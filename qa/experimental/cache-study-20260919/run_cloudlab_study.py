#!/usr/bin/env python3
"""Local SSH orchestration only; all storage, generators and sampling run remotely."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import select
import subprocess
import time
import traceback

HOSTS = ['hp117.utah.cloudlab.us', 'hp118.utah.cloudlab.us', 'hp081.utah.cloudlab.us']
CASES = ['bigdata_baleen_v2', 'graph_graphchi_psw_v2', 'hpc_wrf_continuous_v2',
         'ai_training_ses_v2', 'ai_inference_ses_v2']


class Agent:
    def __init__(self, osd):
        self.osd = osd
        self.process = subprocess.Popen([
            'ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            'wzp@' + HOSTS[osd], 'sudo', '-n', 'python3',
            '/mnt/ceph-lab/cache-study-agent-20260919.py', str(osd)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, action, **fields):
        begin = time.time()
        self.process.stdin.write(json.dumps({'action': action, **fields}) + '\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 180)[0]:
            raise TimeoutError(f'{HOSTS[self.osd]} action {action}')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Agent {self.osd} exited: {self.process.poll()}')
        result = json.loads(line)
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result['data']

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def ready_cluster(cluster):
    s = cluster['status']
    assert s['osdmap']['num_up_osds'] == s['osdmap']['num_in_osds'] == 3
    assert all(p['state_name'] == 'active+clean' for p in s['pgmap']['pgs_by_state'])
    assert set(s['health'].get('checks', {})) <= {'POOL_NO_REDUNDANCY'}, s['health']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent / 'cloudlab-runs' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    agents = [Agent(i) for i in range(3)]
    pool = ThreadPoolExecutor(max_workers=3)

    def all_call(action, **fields):
        return {str(i): result for i, result in enumerate(pool.map(
            lambda agent: agent.call(action, **fields), agents))}

    def stage(case, state, **fields):
        data = {'case': case, 'state': state, 'wall': time.time(), **fields}
        write(root / 'progress.json', data)
        print(json.dumps(data), flush=True)

    try:
        write(root / 'clocks.json', all_call('ping'))
        write(root / 'budget.json', all_call('budget'))
        write(root / 'preflight-samples.json', all_call('sample'))
        runtime = agents[1].call('runtime')
        assert runtime['runtime']['other_methods_identical_after_javap_owner_normalization']
        write(root / 'runtime.json', runtime)
        if args.check_only:
            stage('all', 'preflight-complete-no-measurement')
            return
        while not agents[1].call('preparation')['all_ready']:
            stage('all', 'waiting-for-all-five-persistent-datasets')
            time.sleep(60)
        for case in CASES:
            out = root / case
            out.mkdir()
            initial = agents[0].call('cluster')
            ready_cluster(initial)
            write(out / 'initial-cluster.json', initial)
            write(out / 'baseline-control.json', all_call('control', policy='lru', enabled=False))
            write(out / 'effective-budget.json', all_call('budget'))
            launch = agents[1].call('start', case=case, run_id=args.run_id)
            write(out / 'launch.json', launch)
            timing = {'nominal_seconds': 600, 'switch_target_seconds': 180}
            switched = False
            finished = False
            deadline = time.monotonic() + 900
            count = 0
            with (out / 'samples.jsonl').open('w') as stream:
                while not finished:
                    begin = time.monotonic()
                    row = {'seq': count, 'collection_local_wall': time.time(), 'osds': all_call('sample')}
                    status = agents[1].call('workload')
                    row['workload'] = status
                    if 'start_wall' not in timing and status['phases']:
                        timing['start_wall'] = status['phases'][0]['start_wall']
                        write(out / 'timing.json', timing)
                        stage(case, 'lru-hp-disabled', start_wall=timing['start_wall'])
                    if 'start_wall' in timing:
                        elapsed = status['wall'] - timing['start_wall']
                        if not switched and elapsed >= 179:
                            time.sleep(max(0, 180 - elapsed))
                            controls = all_call('control', policy='s3fifo', enabled=True)
                            write(out / 'switch-control.json', controls)
                            timing['switch_request_wall'] = min(c['requested_wall'] for c in controls.values())
                            timing['switch_confirmed_wall'] = max(c['confirmed_wall'] for c in controls.values())
                            write(out / 'timing.json', timing)
                            switched = True
                            stage(case, 's3fifo-hp-enabled', switch_elapsed=timing['switch_confirmed_wall'] - timing['start_wall'])
                    if count % 10 == 0:
                        row['cluster'] = agents[0].call('cluster')
                        ready_cluster(row['cluster'])
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    count += 1
                    if status['returncode'] is not None:
                        assert status['returncode'] == 0, status
                        assert switched and status['phases'], status
                        expected = runtime['adaptation'][case]['profiles']['current']
                        assert [p['seconds'] for p in status['phases']] == [p['seconds'] for p in expected]
                        timing.update(phases=status['phases'], exit_wall=status['wall'], returncode=0)
                        timing['nominal_end_wall'] = timing['start_wall'] + 600
                        timing['last_rd_nominal_end_wall'] = status['phases'][-1]['start_wall'] + status['phases'][-1]['seconds']
                        write(out / 'timing.json', timing)
                        finished = True
                    if time.monotonic() > deadline:
                        raise TimeoutError('Workload did not finish within 900 seconds')
                    time.sleep(max(0, 2 - (time.monotonic() - begin)))
            write(out / 'workload-complete.json', agents[1].call('finish', case=case))
            drain_deadline = time.monotonic() + 180
            while True:
                samples = all_call('sample')
                if all(s['hp']['hp_pending_io_count'] == s['hp']['hp_awaiting_prediction_count'] ==
                       s['hp']['hp_train_queue_length'] == 0 for s in samples.values()):
                    break
                if time.monotonic() >= drain_deadline:
                    raise TimeoutError('HP queues did not drain')
                time.sleep(2)
            write(out / 'final-samples.json', samples)
            write(out / 'final-cluster.json', agents[0].call('cluster'))
            write(out / 'COMPLETE.json', {'case': case, 'wall': time.time(), 'samples': count})
            stage(case, 'measurement-complete')
        write(root / 'COMPLETE.json', {'cases': CASES, 'wall': time.time()})
    except BaseException:
        (root / 'FAILURE.txt').write_text(traceback.format_exc())
        raise
    finally:
        if not args.check_only:
            try:
                write(root / 'restored-baseline.json', all_call('control', policy='lru', enabled=False))
            except Exception:
                (root / 'RESTORE_FAILURE.txt').write_text(traceback.format_exc())
        pool.shutdown(wait=True)
        for agent in agents:
            agent.close()


if __name__ == '__main__':
    main()
