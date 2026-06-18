# syntax=docker/dockerfile:1
FROM gcr.io/distroless/cc-debian12
COPY target/release/kvcdn /usr/local/bin/kvcdn
ENTRYPOINT ["/usr/local/bin/kvcdn"]
