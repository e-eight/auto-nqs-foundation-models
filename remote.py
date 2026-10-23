# /// script
# requires-python = ">=3.11"
# dependencies = ["websocket-client>=1.7"]
# ///
"""
Run experiments on ICRN. DO NOT MODIFY (the agent only edits tasks/<task>/train.py).

Code moves through GitHub: the current commit is force-pushed to its branch and
checked out on ICRN. Commands run in a Jupyter kernel on an ICRN server started
through the JupyterHub API (H200 slice preferred, A100 slice as fallback).

    uv run remote.py setup [--task j1j2]   # once per task: clone, install, compute references
    uv run remote.py run                   # run HEAD, seed 0 (and HEAD~1 if it has no score)
    uv run remote.py run --seeds 0,1,2     # several seeds, one after another, plus their mean
    uv run remote.py run --seeds 0,1,2 --vary init   # vary only the init seed (MC seed fixed)
    uv run remote.py run --no-baseline
    uv run remote.py status
    uv run remote.py stop

Results are cached per (task, commit, GPU type, init seed, MC seed), so a
confirmation with --seeds 0,1,2 reuses the seed-0 screening run.
"""

import argparse
import json
import math
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".autoresearch"
SCORES = STATE_DIR / "scores.json"
DEFAULT_TASK = "j1j2"
SUMMARY_KEYS = ("status", "reason", "score", "mean_rel_err", "max_rel_err", "worst_j2", "probe_ratio",
                "training_seconds", "total_seconds", "num_steps", "num_params", "peak_vram_gb")
# Exit codes of harness.py; any other non-zero exit is a crash.
EXIT_STATUS = {2: "invalid", 3: "diverged"}
# Results worth caching: they depend on the code and seed, not on the infrastructure.
CACHED_STATUSES = ("ok", "invalid", "diverged", "crash")


def log(msg):
    print(f"[remote {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Fail(Exception):
    pass


# ---------------------------------------------------------------------------
# JupyterHub + Jupyter server access
# ---------------------------------------------------------------------------


class Hub:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = cfg["hub_url"].rstrip("/")
        self.user = cfg["user"]
        self.name = cfg["server"]
        self._token = Path(cfg["token_file"]).expanduser().read_text().strip()
        self.kernel_id = None

    def _req(self, method, path, body=None, ok=(200, 201, 202, 204)):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"token {self._token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return r.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if e.code in ok:
                return e.code, None
            raise Fail(f"{method} {path} -> HTTP {e.code}: {e.read()[:300]!r}") from None

    @property
    def _server_api(self):
        suffix = f"servers/{self.name}" if self.name else "server"
        return f"/hub/api/users/{self.user}/{suffix}"

    def server(self):
        _, info = self._req("GET", f"/hub/api/users/{self.user}")
        return info["servers"].get(self.name)

    def start(self, user_options):
        self._req("POST", self._server_api, body=user_options)

    def stop(self):
        self._req("DELETE", self._server_api, ok=(202, 204, 404))
        for _ in range(60):
            if self.server() is None:
                return
            time.sleep(5)
        raise Fail("server did not stop")

    # -- kernel ------------------------------------------------------------

    def _server_url(self):
        s = self.server()
        if not s or not s.get("ready"):
            raise Fail("server not ready")
        return s["url"]

    def _kernel(self):
        url = self._server_url()
        if self.kernel_id:
            try:
                self._req("GET", f"{url}api/kernels/{self.kernel_id}")
                return url, self.kernel_id
            except Fail:
                self.kernel_id = None
        _, k = self._req("POST", f"{url}api/kernels", body={"name": "python3"})
        self.kernel_id = k["id"]
        return url, self.kernel_id

    def py(self, code, timeout=180):
        """Execute Python in a kernel on the server; return captured stdout/stderr."""
        url, kid = self._kernel()
        ws_url = self.base.replace("https://", "wss://").replace("http://", "ws://")
        ws = websocket.create_connection(
            f"{ws_url}{url}api/kernels/{kid}/channels",
            header=[f"Authorization: token {self._token}"],
            timeout=timeout,
        )
        msg_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": self.user, "session": uuid.uuid4().hex,
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        }))
        out, err, deadline = [], None, time.time() + timeout
        try:
            while time.time() < deadline:
                msg = json.loads(ws.recv())
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    out.append(content["text"])
                elif kind == "error":
                    err = f"{content['ename']}: {content['evalue']}"
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            else:
                raise Fail("kernel execution timed out")
        finally:
            ws.close()
        if err:
            raise Fail(f"remote error: {err}\n{''.join(out)[-2000:]}")
        return "".join(out)

    def sh(self, script, timeout=180):
        """Run a bash script on the server; raise on non-zero exit."""
        code = (
            "import subprocess, sys\n"
            f"r = subprocess.run(['bash', '-lc', {script!r}], capture_output=True, text=True)\n"
            "print(r.stdout[-20000:], end='')\n"
            "print(r.stderr[-20000:], end='', file=sys.stderr)\n"
            "print(f'\\n__EXIT__={r.returncode}')\n"
        )
        out = self.py(code, timeout=timeout)
        body, _, code_s = out.rpartition("__EXIT__=")
        if code_s.strip() != "0":
            raise Fail(f"remote command failed (exit {code_s.strip()}):\n{body[-3000:]}")
        return body.strip()


# ---------------------------------------------------------------------------
# Server lifecycle: H200 first, A100 fallback
# ---------------------------------------------------------------------------


def gpu_class(hub):
    out = hub.sh("nvidia-smi -L 2>/dev/null || true")
    for name in ("H200", "H100", "A100"):
        if name in out:
            return name.lower()
    return None


def wait_ready(hub, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        s = hub.server()
        if s is None:
            return False  # spawn failed and the server was removed
        if s.get("ready"):
            return True
        time.sleep(10)
    return False


def ensure_gpu_server(hub, cfg):
    """Make sure the experiment server is up on a GPU; return its GPU class."""
    t = cfg["timing"]
    for r in cfg["resources"]:
        if "FILL_ME" in json.dumps(r["user_options"]):
            raise Fail("set the GPU resource values in icrn.toml first")
    s = hub.server()
    if s and s.get("pending") == "spawn":
        log("server is still starting; waiting")
        wait_ready(hub, t["spawn_wait"])
        s = hub.server()
    if s and s.get("pending") is None and s.get("ready"):
        gpu = gpu_class(hub)
        if gpu:
            return gpu
        log("server is running without a GPU; restarting it")
    if s:
        hub.stop()
    queue_start = time.time()
    while True:
        for r in cfg["resources"]:
            log(f"starting server '{hub.name}' on {r['gpu']}")
            try:
                hub.start(r["user_options"])
            except Fail as e:
                log(f"spawn request failed: {e}")
                continue
            if wait_ready(hub, t["spawn_wait"]):
                gpu = gpu_class(hub)
                if gpu:
                    log(f"server ready on {gpu}")
                    return gpu
                log("server came up without a visible GPU")
            log(f"{r['gpu']} not available")
            if hub.server():
                hub.stop()
        if time.time() - queue_start > t["max_queue_wait"]:
            raise Fail("no GPU server could be started")
        log(f"no GPU free; retrying in {t['retry_wait']}s")
        time.sleep(t["retry_wait"])


# ---------------------------------------------------------------------------
# Git + remote jobs
# ---------------------------------------------------------------------------


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def push_head():
    if git("status", "--porcelain", "--untracked-files=no"):
        raise Fail("uncommitted changes to tracked files; commit train.py first")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise Fail("detached HEAD; check out the experiment branch")
    git("push", "--quiet", "--force", "origin", f"HEAD:refs/heads/{branch}")
    return branch


def remote_prelude(cfg, hub):
    """Bash that locates gh/uv and enters this server's checkout."""
    work = f"{cfg['remote_root']}/{hub.name or 'default'}"
    return f"""
set -euo pipefail
export UV_LINK_MODE=copy
GH={cfg['gh']}
UV=$(command -v uv || ls ~/.local/bin/uv ~/.pixi/bin/uv 2>/dev/null | head -1 || true)
if [ -z "$UV" ]; then
  [ -x ~/.pixi/bin/pixi ] || {{ echo "uv not found on ICRN and no pixi to install it"; exit 1; }}
  ~/.pixi/bin/pixi global install uv >/dev/null && UV=~/.pixi/bin/uv
fi
WORK={work}
# HTTPS with gh as credential helper: does not depend on SSH keys or known_hosts.
git_auth() {{ git -c credential.helper= -c "credential.helper=!$GH auth git-credential" "$@"; }}
if [ ! -d "$WORK/.git" ]; then
  rm -rf "$WORK"; mkdir -p {cfg['remote_root']}
  git_auth clone --quiet https://github.com/{cfg['repo']}.git "$WORK"
fi
cd "$WORK"
git remote set-url origin https://github.com/{cfg['repo']}.git
"""


def checkout(sha, branch):
    return f"""
git_auth fetch --quiet origin +refs/heads/{branch}:refs/remotes/origin/{branch} || true
git cat-file -e {sha}^{{commit}} 2>/dev/null || git_auth fetch --quiet origin {sha}
git checkout --quiet --detach {sha}
"""


def launch(hub, cfg, script, tag):
    """Start `script` detached on the server; return the remote run directory."""
    rundir = f"{cfg['remote_root']}/runs/{tag}"
    job = f"{remote_prelude(cfg, hub)}\n{script}"
    hub.sh(f"""
mkdir -p {rundir} && cd {rundir} && rm -f exit run.log
cat > job.sh <<'__JOB__'
{job}
__JOB__
setsid nohup bash -c 'bash {rundir}/job.sh > {rundir}/run.log 2>&1; echo $? > {rundir}/exit' >/dev/null 2>&1 &
echo $! > pid
""")
    return rundir


def wait_job(hub, cfg, rundir):
    """Poll a detached job; return (exit_code or None on timeout, log text)."""
    t = cfg["timing"]
    start = time.time()
    while True:
        time.sleep(t["poll_every"])
        done = hub.sh(f"cat {rundir}/exit 2>/dev/null || true")
        if done:
            return int(done), hub.sh(f"tail -c 200000 {rundir}/run.log")
        if time.time() - start > t["run_timeout"]:
            hub.sh(f"kill -TERM -- -$(cat {rundir}/pid) 2>/dev/null || true")
            return None, hub.sh(f"tail -c 200000 {rundir}/run.log")


def parse_summary(text):
    out = {}
    for line in text.splitlines():
        key, sep, val = line.partition(": ")
        if sep and key in SUMMARY_KEYS:
            out[key] = val.strip()
    return out


def load_scores():
    return json.loads(SCORES.read_text()) if SCORES.exists() else {}


def save_score(key, rec):
    scores = load_scores()
    scores[key] = rec
    STATE_DIR.mkdir(exist_ok=True)
    tmp = SCORES.with_suffix(".tmp")
    tmp.write_text(json.dumps(scores, indent=1))
    tmp.replace(SCORES)


def classify(code, summary):
    """Turn the exit code and parsed summary of a run into a trustworthy status.

    A run only counts as `ok` if it exited cleanly and printed a finite score;
    a NaN or missing score is `diverged`, never a number to compare against.
    """
    if code is None:
        return {"status": "timeout"}
    status = summary.get("status")
    if code == 0 and status == "ok":
        try:
            score = float(summary.get("score", "nan"))
        except ValueError:
            score = math.nan
        if math.isfinite(score):
            return summary
        return {**summary, "status": "diverged", "reason": f"non-finite score ({summary.get('score')})"}
    if code in EXIT_STATUS and status == EXIT_STATUS[code]:
        return summary
    return {"status": "crash"}


def run_commit(hub, cfg, sha, branch, gpu, task, seed, mc_seed):
    """Run tasks/<task>/train.py at `sha` with the given seeds; return the summary dict."""
    tag = f"{task}-{sha[:7]}-{gpu}-s{seed}m{mc_seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    script = checkout(sha, branch) + (
        '"$UV" sync --frozen --extra cuda --quiet\n'
        f'NQS_SEED={seed} NQS_MC_SEED={mc_seed} PYTHONPATH="$WORK" '
        f'"$UV" run --frozen --extra cuda tasks/{task}/train.py\n'
    )
    log(f"running {task} {sha[:7]} on {gpu} (seed {seed}, mc seed {mc_seed})")
    rundir = launch(hub, cfg, script, tag)
    code, text = wait_job(hub, cfg, rundir)
    (STATE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "logs" / f"{tag}.log").write_text(text)
    summary = classify(code, parse_summary(text))
    summary["log"] = str((STATE_DIR / "logs" / f"{tag}.log").relative_to(ROOT))
    if summary.get("status") != "ok":
        print("--- remote log tail ---")
        print("\n".join(text.splitlines()[-60:]))
    return summary


def score_key(task, sha, gpu, seed, mc_seed):
    return f"{task}:{sha}:{gpu}:{seed}:{mc_seed}"


def scored(hub, cfg, sha, branch, gpu, task, seed, mc_seed, fresh=False):
    """Cached result of (sha, gpu, seeds), running it if needed."""
    key = score_key(task, sha, gpu, seed, mc_seed)
    cached = load_scores().get(key)
    if cached is not None and not fresh:
        log(f"cached: {task} {sha[:7]} on {gpu} (seed {seed}, mc seed {mc_seed}): {cached.get('status')}")
        return cached
    summary = run_commit(hub, cfg, sha, branch, gpu, task, seed, mc_seed)
    if summary.get("status") in CACHED_STATUSES:
        save_score(key, summary)
    return summary


def aggregate(results):
    """Mean and worst score over seeds; status is `ok` only if every seed is."""
    bad = [r.get("status") for r in results if r.get("status") != "ok"]
    scores = [float(r["score"]) for r in results if r.get("status") == "ok"]
    errs = [float(r["max_rel_err"]) for r in results if r.get("status") == "ok" and "max_rel_err" in r]
    out = {"status": bad[0] if bad else "ok", "seeds_ok": f"{len(scores)}/{len(results)}"}
    if scores:
        out["score"] = f"{sum(scores) / len(scores):.6f}"
        out["worst_score"] = f"{max(scores):.6f}"
        out["best_score"] = f"{min(scores):.6f}"
    if errs:
        out["max_rel_err"] = f"{max(errs):.6f}"
    return out


def seed_pairs(seeds, vary):
    """(init seed, MC seed) per run. `both` mirrors harness.seeds() defaults."""
    if vary == "init":
        return [(s, 10_000) for s in seeds]
    if vary == "mc":
        return [(0, 10_000 + s) for s in seeds]
    return [(s, 10_000 + s) for s in seeds]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(hub, cfg, args):
    branch = push_head()
    sha = git("rev-parse", "HEAD")
    parent = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "HEAD~1"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()
    pairs = seed_pairs(args.seeds, args.vary)
    for attempt in (1, 2):
        try:
            gpu = ensure_gpu_server(hub, cfg)
            baseline = None
            if parent and not args.no_baseline:
                baseline = [scored(hub, cfg, parent, branch, gpu, args.task, s, m) for s, m in pairs]
            results = [scored(hub, cfg, sha, branch, gpu, args.task, s, m, fresh=args.fresh)
                       for s, m in pairs]
            break
        except (Fail, websocket.WebSocketException, OSError) as e:
            if attempt == 2:
                raise
            log(f"lost the server ({e}); retrying once")
    print("---")
    print(f"task: {args.task}")
    print(f"commit: {sha[:7]}")
    print(f"gpu: {gpu}")
    if len(results) == 1:
        for k in SUMMARY_KEYS:
            if k in results[0]:
                print(f"{k}: {results[0][k]}")
    else:
        print(f"seeds: {','.join(f'{s}/{m}' for s, m in pairs)}")
        for k, v in aggregate(results).items():
            print(f"{k}: {v}")
        print("seed_scores: " + " ".join(
            f"{s}/{m}={r.get('score', r.get('status'))}" for (s, m), r in zip(pairs, results)))
    if baseline is not None:
        agg = aggregate(baseline)
        print(f"baseline_commit: {parent[:7]}")
        print(f"baseline_status: {agg['status']}")
        # A baseline that is not ok on every seed has no score to compare against.
        print(f"baseline_score: {agg['score'] if agg['status'] == 'ok' else 'none'}")
        if len(baseline) > 1 and agg["status"] == "ok":
            print(f"baseline_worst_score: {agg['worst_score']}")
    print("log: " + " ".join(r["log"] for r in results if "log" in r))


def cmd_setup(hub, cfg, args):
    branch = push_head()
    sha = git("rev-parse", "HEAD")
    gpu = ensure_gpu_server(hub, cfg)
    script = checkout(sha, branch) + (
        '"$UV" sync --frozen --extra cuda\n'
        f'"$UV" run --frozen --extra cuda prepare.py --task {args.task}\n'
    )
    rundir = launch(hub, cfg, script, f"setup-{args.task}-{gpu}-{time.strftime('%Y%m%d-%H%M%S')}")
    log("installing environment and computing reference energies")
    code, text = wait_job(hub, cfg, rundir)
    print("\n".join(text.splitlines()[-40:]))
    print("---")
    print(f"setup: {'ok' if code == 0 else 'failed'}")
    if code != 0:
        sys.exit(1)


def cmd_status(hub, cfg, args):
    s = hub.server()
    print(json.dumps(s, indent=1) if s else f"server '{hub.name}' is not running")
    if s and s.get("ready"):
        print(f"gpu: {gpu_class(hub)}")


def cmd_stop(hub, cfg, args):
    hub.stop()
    print(f"server '{hub.name}' stopped")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["setup", "run", "status", "stop"])
    p.add_argument("--task", default=DEFAULT_TASK, help="task directory under tasks/")
    p.add_argument("--seeds", default=[0], type=lambda v: [int(x) for x in v.split(",")],
                   help="comma-separated seeds, run one after another (default 0)")
    p.add_argument("--vary", choices=["both", "init", "mc"], default="both",
                   help="which seed --seeds varies: network init, Monte Carlo, or both")
    p.add_argument("--no-baseline", action="store_true", help="do not (re)score HEAD~1 first")
    p.add_argument("--fresh", action="store_true", help="re-run HEAD even if its result is cached")
    p.add_argument("--server", help="override the server name from icrn.toml")
    args = p.parse_args()
    cfg = tomllib.loads((ROOT / "icrn.toml").read_text())
    if args.server is not None:
        cfg["server"] = args.server
    hub = Hub(cfg)
    try:
        {"setup": cmd_setup, "run": cmd_run, "status": cmd_status, "stop": cmd_stop}[args.command](hub, cfg, args)
    except (Fail, websocket.WebSocketException, OSError) as e:
        print(f"---\nstatus: remote_error\nreason: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
