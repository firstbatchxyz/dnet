import os

GRPC_MAX_MESSAGE_LENGTH = int(os.getenv("GRPC_MAX_MESSAGE_LENGTH", 64 * 1024 * 1024))
GRPC_AIO_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    # Concurrency for streaming
    ("grpc.max_concurrent_streams", 1024),
    # Conservative keepalive/pings to avoid GOAWAY too_many_pings
    ("grpc.keepalive_time_ms", 120000),  # 2 minutes between pings
    ("grpc.keepalive_timeout_ms", 20000),  # 20s timeout for keepalive
    ("grpc.keepalive_permit_without_calls", 0),  # no pings without calls
    ("grpc.http2.min_time_between_pings_ms", 120000),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.bdp_probe", 0),  # disable BDP probe to reduce pinging
    # Avoid any interference from HTTP proxies for direct ring links
    ("grpc.enable_http_proxy", 0),
]
