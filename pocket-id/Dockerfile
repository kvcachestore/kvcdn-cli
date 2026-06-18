FROM ghcr.io/pocket-id/pocket-id:latest

USER root

RUN apk add --no-cache sqlite

COPY entrypoint.sh /usr/local/bin/kvcdn-pocket-id-entrypoint.sh
RUN chmod +x /usr/local/bin/kvcdn-pocket-id-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/kvcdn-pocket-id-entrypoint.sh"]
CMD ["/app/pocket-id"]
