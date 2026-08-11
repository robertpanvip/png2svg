// 用 @resvg/resvg-js（resvg 编译为 WASM）把 SVG 渲染成 PNG。
// 用法: node render_svg.js <in.svg> <out.png> [zoom]
// 依赖: NODE_PATH 指向托管 node workspace 的 node_modules
const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');

const args = process.argv.slice(2);
if (args.length < 2) {
  console.error('usage: render_svg.js <in.svg> <out.png> [zoom]');
  process.exit(1);
}
const inSvg = args[0];
const outPng = args[1];
const zoom = args[2] ? parseFloat(args[2]) : 1;

const svg = fs.readFileSync(inSvg, 'utf8');
const opts = {
  fitTo: { mode: 'zoom', value: zoom },
  background: 'rgba(0,0,0,0)', // 透明背景，保留 alpha
  logLevel: 'off',
};
const resvg = new Resvg(svg, opts);
const png = resvg.render();
fs.writeFileSync(outPng, png.asPng());
console.log('rendered ' + inSvg + ' -> ' + outPng + ' (' + Math.round(200 * zoom) + 'px @zoom ' + zoom + ')');
