require('ts-node').register({
  project: 'tsconfig.json'
});
require('tsconfig-paths').register({
  baseUrl: './',
  paths: {
    '*': ['src/*']
  }
});