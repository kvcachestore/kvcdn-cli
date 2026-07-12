FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev
COPY dist ./dist
EXPOSE 3000
CMD ["node", "dist/index.js"]
