import socket
import ssl
import sys

HOSTS = [
    ("pypi.org", 443),
    ("files.pythonhosted.org", 443),
    ("pypi.tuna.tsinghua.edu.cn", 443),
    ("mirrors.aliyun.com", 443),
    ("push2his.eastmoney.com", 443),
    ("www.baidu.com", 443),
]

for host, port in HOSTS:
    try:
        s = socket.create_connection((host, port), timeout=6)
        s.close()
        tcp = "TCP_OK"
    except Exception as exc:  # noqa: BLE001
        print(f"{host:35s} TCP_FAIL {type(exc).__name__}: {exc}")
        continue
    try:
        ctx = ssl.create_default_context()
        raw = socket.create_connection((host, port), timeout=6)
        with ctx.wrap_socket(raw, server_hostname=host) as ss:
            print(f"{host:35s} {tcp} TLS_OK")
    except Exception as exc:  # noqa: BLE001
        print(f"{host:35s} {tcp} TLS_FAIL {type(exc).__name__}: {exc}")

print("python", sys.version)
