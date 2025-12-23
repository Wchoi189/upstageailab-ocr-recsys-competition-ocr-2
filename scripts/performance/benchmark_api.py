
import time
import base64
import urllib.request
import urllib.error
import json
import sys
import sys

API_URL = "http://localhost:8002/api/inference/preview"
HEALTH_URL = "http://localhost:8002/api/health"
IMAGE_PATH = "tests/e2e/fixtures/sample-image.jpg"

def main():
    # 1. Wait for health
    print("Waiting for server...")
    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(HEALTH_URL) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    if data.get("engine_loaded"):
                        print("Server ready.")
                        ready = True
                        break
                    else:
                        print("Server up but engine not loaded yet...")
        except (urllib.error.URLError, ConnectionResetError):
            print("Waiting for connection...")
            pass
        time.sleep(1)

    if not ready:
        print("Server failed to allow connection or load engine.")
        sys.exit(1)

    # 2. Prepare request
    print(f"Loading image from {IMAGE_PATH}")
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "checkpoint_path": "", # Use latest
        "image_base64": img_b64,
        "confidence_threshold": 0.5,
        "nms_threshold": 0.5
    }

    data_bytes = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    # 3. Warmup
    print("Warming up (5 requests)...")
    for i in range(5):
        try:
            req = urllib.request.Request(API_URL, data=data_bytes, headers=headers)
            with urllib.request.urlopen(req) as r:
                if r.status != 200:
                    print(f"Warmup failed: {r.status}")
        except Exception as e:
            print(f"Warmup error: {e}")

    # 4. Benchmark
    latencies = []
    processing_times = []
    N = 20
    print(f"Running {N} requests...")

    for i in range(N):
        start = time.perf_counter()
        try:
            req = urllib.request.Request(API_URL, data=data_bytes, headers=headers)
            with urllib.request.urlopen(req) as resp:
                elapsed = (time.perf_counter() - start) * 1000

                response_body = resp.read()
                data = json.loads(response_body)
                server_time = data.get("processing_time_ms", 0)

                latencies.append(elapsed)
                processing_times.append(server_time)
                print(f"Req {i+1}: Total={elapsed:.2f}ms, Server={server_time:.2f}ms")
        except urllib.error.HTTPError as e:
             print(f"HTTP Error: {e.code} - {e.read()}")
        except Exception as e:
            print(f"Request error: {e}")

        time.sleep(0.1)

    if not latencies:
        print("No successful requests.")
        sys.exit(1)

    latencies.sort()
    processing_times.sort()

    def get_p(data, p):
        if not data: return 0
        return data[int(len(data) * p)]

    p50_total = get_p(latencies, 0.50)
    p90_total = get_p(latencies, 0.90)
    p99_total = get_p(latencies, 0.99)

    p50_server = get_p(processing_times, 0.50)
    p90_server = get_p(processing_times, 0.90)

    print(f"\nResults (Total Latency):")
    print(f"P50: {p50_total:.2f}ms")
    print(f"P90: {p90_total:.2f}ms")
    print(f"P99: {p99_total:.2f}ms")

    print(f"\nResults (Server Processing Time):")
    print(f"P50: {p50_server:.2f}ms")
    print(f"P90: {p90_server:.2f}ms")

    results = {
        "p50_latency": p50_total,
        "p90_latency": p90_total,
        "p99_latency": p99_total,
        "p50_processing": p50_server,
        "throughput_est": 1000 / p50_total
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark_results.json")

if __name__ == "__main__":
    main()

