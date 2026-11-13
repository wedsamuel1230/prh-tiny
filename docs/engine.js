/* The same model the firmware runs, in the browser: int8 weights dequantised once at load,
   float activations, identical layer order. Used by the demo page and by the design mockups. */

const b64ToBytes = (s) => {
  const bin = atob(s);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
};
const f32 = (s) => new Float32Array(b64ToBytes(s).buffer);
const i8 = (s) => new Int8Array(b64ToBytes(s).buffer);

async function loadModel(url) {
  const m = await fetch(url).then((r) => r.json());
  m.convW = m.convs.map((c) => {
    const q = i8(c.w);
    const s = f32(c.s);
    const w = new Float32Array(q.length);
    const per = 9 * c.ic;
    for (let o = 0; o < c.oc; o++)
      for (let k = 0; k < per; k++) w[o * per + k] = q[o * per + k] * s[o];
    return { w, b: f32(c.b), ic: c.ic, oc: c.oc };
  });
  {
    const q = i8(m.fc.w), s = f32(m.fc.s), w = new Float32Array(q.length);
    for (let o = 0; o < q.length / m.d_img; o++)
      for (let k = 0; k < m.d_img; k++) w[o * m.d_img + k] = q[o * m.d_img + k] * s[o];
    m.fcW = { w, b: f32(m.fc.b) };
  }
  m.emb = f32(m.txt.emb);
  m.fc1 = { w: f32(m.txt.fc1.w), b: f32(m.txt.fc1.b) };
  m.fc2 = { w: f32(m.txt.fc2.w), b: f32(m.txt.fc2.b) };
  m.R = i8(m.rot.R);
  m.Rs = f32(m.rot.s);
  m.muA = f32(m.rot.mu_a);
  m.muB = f32(m.rot.mu_b);
  m.proto = f32(m.protos);
  m.word = new Map(Object.entries(m.txt.vocab));
  return m;
}

function conv3x3s2(x, H, W, layer, out) {
  const { w, b, ic, oc } = layer;
  const OH = H >> 1, OW = W >> 1;
  const acc = new Float32Array(oc);
  for (let oy = 0; oy < OH; oy++)
    for (let ox = 0; ox < OW; ox++) {
      acc.fill(0);
      for (let ky = 0; ky < 3; ky++) {
        const iy = oy * 2 + ky - 1;
        if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 3; kx++) {
          const ix = ox * 2 + kx - 1;
          if (ix < 0 || ix >= W) continue;
          const px = (iy * W + ix) * ic;
          for (let o = 0; o < oc; o++) {
            const base = ((o * 9 + ky * 3 + kx) * ic) | 0;
            let s = 0;
            for (let c = 0; c < ic; c++) s += x[px + c] * w[base + c];
            acc[o] += s;
          }
        }
      }
      const dst = (oy * OW + ox) * oc;
      for (let o = 0; o < oc; o++) out[dst + o] = Math.max(0, acc[o] + b[o]);
    }
  return [OH, OW];
}

/* rgb: Uint8ClampedArray-like, m.img*m.img*3, RGB888. out: Float32Array(d_img). */
function embedImage(m, rgb) {
  const S = m.img;
  let H = S, W = S, IC = 3;
  /* every layer's output must fit: the first one (48x48x24) is larger than the input */
  const cap = Math.max(S * S * 3,
    ...m.channels.slice(1).map((c, i) => ((S >> (i + 1)) ** 2) * c));
  let a = new Float32Array(cap);
  let buf = new Float32Array(cap);
  for (let i = 0; i < S * S * 3; i++) a[i] = rgb[i] / 128 - 1;
  for (const L of m.convW) {
    [H, W] = conv3x3s2(a, H, W, L, buf);
    [a, buf] = [buf, a];
    IC = L.oc;
  }
  const pool = new Float32Array(IC);
  for (let i = 0; i < H * W; i++) for (let c = 0; c < IC; c++) pool[c] += a[i * IC + c];
  const k = H * W;
  const out = new Float32Array(m.d_img);
  const { w, b } = m.fcW;
  for (let o = 0; o < m.d_img; o++) {
    let s = 0;
    for (let c = 0; c < IC; c++) s += (pool[c] / k) * w[o * IC + c];
    out[o] = s + b[o];
  }
  return out;
}

function rotate(m, emb) {
  const da = m.d_img, db = m.d_txt;
  const y = new Float32Array(db);
  for (let i = 0; i < da; i++) {
    const qf = (emb[i] - m.muA[i]) * m.Rs[i];
    const row = i * db;
    for (let j = 0; j < db; j++) y[j] += qf * m.R[row + j];
  }
  for (let j = 0; j < db; j++) y[j] += m.muB[j];
  return y;
}

function encodeText(m, sentence) {
  const ids = sentence.toLowerCase().split(/[^a-z]+/).filter(Boolean)
    .map((w) => m.word.get(w)).filter((i) => i !== undefined);
  if (!ids.length) return null;
  const d = m.txt.d_emb;
  const h = new Float32Array(d);
  for (const id of ids) for (let c = 0; c < d; c++) h[c] += m.emb[id * d + c];
  for (let c = 0; c < d; c++) h[c] /= ids.length;

  const h1 = new Float32Array(128);
  for (let o = 0; o < 128; o++) {
    let s = 0;
    for (let c = 0; c < d; c++) s += h[c] * m.fc1.w[o * d + c];
    h1[o] = Math.max(0, s + m.fc1.b[o]);
  }
  const out = new Float32Array(m.d_txt);
  for (let o = 0; o < m.d_txt; o++) {
    let s = 0;
    for (let c = 0; c < 128; c++) s += h1[c] * m.fc2.w[o * 128 + c];
    out[o] = s + m.fc2.b[o];
  }
  return out;
}

const cos = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
};

function rank(m, y) {
  const d = m.d_txt;
  return m.names
    .map((name, i) => ({ name, score: cos(y, m.proto.subarray(i * d, (i + 1) * d)) }))
    .sort((a, b) => b.score - a.score);
}

function demoFrame(m, i) {
  return b64ToBytes(m.demos[i].rgb);
}
