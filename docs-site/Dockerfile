# Static docs site (API reference + mdBook), built by build.sh into dist/
# before this image is built. Mirrors ../../spinorml-cdn's Dockerfile
# (nginx serving static files from /data, no application logic).
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY dist/ /data/

EXPOSE 80
