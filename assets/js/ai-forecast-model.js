// ECMWF features, METAR verification, and a deterministic random-forest residual model.
export const TARGETS = [
  { key: 'low', label: 'Minimum temperature', unit: '°F', digits: 1 },
  { key: 'high', label: 'Maximum temperature', unit: '°F', digits: 1 },
  { key: 'rain', label: 'Precipitation', unit: 'in', digits: 2 },
  { key: 'gust', label: 'Maximum reported wind gust', unit: 'kt', digits: 1 }
];
export const INPUTS = [
  { key: 'rhMin', label: 'Minimum relative humidity', unit: '%', digits: 0 },
  { key: 'rhMax', label: 'Maximum relative humidity', unit: '%', digits: 0 },
  { key: 'cloudMean', label: 'Mean cloud cover', unit: '%', digits: 0 },
  { key: 'capeMean', label: 'Mean CAPE', unit: 'J/kg', digits: 0 },
  { key: 'direction', label: 'Dominant wind direction', unit: '°', digits: 0 },
  { key: 'windMax', label: 'Maximum wind speed', unit: 'kt', digits: 1 }
];
export const HOURLY_FIELDS = 'temperature_2m,precipitation,wind_gusts_10m,relative_humidity_2m,cloud_cover,cape,wind_direction_10m,wind_speed_10m';
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
export function shiftDay(day, offset) {
  const date = new Date(`${day}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}
export function cycleBounds(day) {
  const start = Date.parse(`${day}T06:00:00Z`) / 1000;
  return { start, end: start + 86400 };
}
export function aggregateForecast(hourly, day) {
  const { start } = cycleBounds(day);
  const index = new Map((hourly?.time || []).map((time, i) => [time, i]));
  const hours = Array.from({ length: 24 }, (_, i) => start + i * 3600);
  const take = (field, offset = 0) => hours.map(time => hourly?.[field]?.[index.get(time + offset)]);
  const complete = values => values.length === 24 && values.every(finite);
  const extrema = (values, type) => complete(values) ? Math[type](...values) : null;
  const average = values => complete(values) ? mean(values) : null;
  const temperature = take('temperature_2m');
  const precipitation = take('precipitation', 3600); // preceding-hour amounts
  const gust = take('wind_gusts_10m');
  const direction = take('wind_direction_10m');
  const speed = take('wind_speed_10m');
  let dominant = null;
  if (complete(direction) && complete(speed)) {
    const radians = direction.map(value => value * Math.PI / 180);
    const x = mean(radians.map((angle, i) => Math.cos(angle) * speed[i]));
    const y = mean(radians.map((angle, i) => Math.sin(angle) * speed[i]));
    if (Math.hypot(x, y) > 1e-9) dominant = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }
  return {
    low: extrema(temperature, 'min'), high: extrema(temperature, 'max'),
    rain: complete(precipitation) ? precipitation.reduce((sum, value) => sum + value, 0) : null,
    gust: extrema(gust, 'max'),
    rhMin: extrema(take('relative_humidity_2m'), 'min'),
    rhMax: extrema(take('relative_humidity_2m'), 'max'),
    cloudMean: average(take('cloud_cover')),
    capeMean: average(take('cape')),
    direction: dominant, windMax: extrema(speed, 'max')
  };
}
export function parseAsosCSV(csv) {
  const lines = csv.trim().split(/\r?\n/);
  const header = lines.shift()?.split(',');
  if (!header || !['station', 'valid', 'tmpf', 'gust', 'sknt', 'p01i'].every(field => header.includes(field))) throw new Error('Unexpected station observation format');
  return lines.filter(line => line && !line.startsWith('#')).map(line => {
    const cells = line.split(',');
    const read = field => cells[header.indexOf(field)];
    const numeric = field => {
      const value = read(field);
      if (value === 'T') return 0; // trace below reporting resolution
      if (value === '' || value === 'M') return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    };
    const time = Date.parse(`${read('valid').replace(' ', 'T')}:00Z`) / 1000;
    return { time, tmpf: numeric('tmpf'), gust: numeric('gust'), sknt: numeric('sknt'), p01i: numeric('p01i') };
  }).filter(row => finite(row.time));
}
export function aggregateObservations(rows, day) {
  const { start, end } = cycleBounds(day);
  const reports = rows.filter(row => row.time >= start && row.time < end);
  const countHours = field => new Set(reports.filter(row => finite(row[field])).map(row => Math.floor(row.time / 3600))).size;
  const values = field => reports.map(row => row[field]).filter(finite);
  const temp = values('tmpf');
  const rain = rows.filter(row => row.time > start && row.time <= end).map(row => row.p01i).filter(finite);
  const gust = values('gust');
  return {
    low: countHours('tmpf') >= 18 ? Math.min(...temp) : null,
    high: countHours('tmpf') >= 18 ? Math.max(...temp) : null,
    rain: countHours('p01i') >= 18 && rain.length >= 18 ? rain.reduce((sum, value) => sum + value, 0) : null,
    gust: countHours('sknt') >= 18 ? (gust.length ? Math.max(...gust) : 0) : null
  };
}
export function features(raw) {
  if (!raw) return null;
  const direction = raw.direction;
  const vector = [raw.low, raw.high, raw.rain, raw.gust, raw.rhMin, raw.rhMax,
    raw.cloudMean, raw.capeMean, raw.windMax,
    finite(direction) ? Math.sin(direction * Math.PI / 180) : null,
    finite(direction) ? Math.cos(direction * Math.PI / 180) : null];
  return vector.every(finite) ? vector : null;
}
export const FOREST_SETTINGS = Object.freeze({ trees: 100, maxDepth: 4, minSamplesLeaf: 5, maxFeatures: 'sqrt' });
function random(seed) {
  let state = seed >>> 0;
  return () => ((state = (Math.imul(1664525, state) + 1013904223) >>> 0) / 4294967296);
}
function tree(rows, depth, rng, dimensions) {
  const average = mean(rows.map(row => row.y));
  if (depth >= FOREST_SETTINGS.maxDepth || rows.length < FOREST_SETTINGS.minSamplesLeaf * 2) return { value: average };
  const candidates = Array.from({ length: dimensions }, (_, i) => i);
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }
  const featureCount = Math.max(1, Math.floor(Math.sqrt(dimensions)));
  let best = null;
  for (const feature of candidates.slice(0, featureCount)) {
    const sorted = rows.slice().sort((a, b) => a.x[feature] - b.x[feature]);
    let leftSum = 0, leftSquares = 0;
    const totalSum = sorted.reduce((sum, row) => sum + row.y, 0);
    const totalSquares = sorted.reduce((sum, row) => sum + row.y * row.y, 0);
    for (let i = 1; i < sorted.length; i++) {
      const y = sorted[i - 1].y;
      leftSum += y; leftSquares += y * y;
      if (i < FOREST_SETTINGS.minSamplesLeaf || sorted.length - i < FOREST_SETTINGS.minSamplesLeaf || sorted[i - 1].x[feature] === sorted[i].x[feature]) continue;
      const rightSum = totalSum - leftSum;
      const loss = leftSquares - leftSum * leftSum / i + totalSquares - leftSquares - rightSum * rightSum / (sorted.length - i);
      if (!best || loss < best.loss) best = { loss, feature, threshold: (sorted[i - 1].x[feature] + sorted[i].x[feature]) / 2 };
    }
  }
  if (!best) return { value: average };
  const left = rows.filter(row => row.x[best.feature] < best.threshold);
  const right = rows.filter(row => row.x[best.feature] >= best.threshold);
  return { feature: best.feature, threshold: best.threshold,
    left: tree(left, depth + 1, rng, dimensions), right: tree(right, depth + 1, rng, dimensions) };
}
export function trainForest(rows, seed = 2026) {
  if (rows.length < 30) return null;
  const dimensions = rows[0].x.length;
  if (!rows.every(row => row.x.length === dimensions && row.x.every(finite) && finite(row.y))) throw new Error('Invalid forest training row');
  const rng = random(seed);
  const trees = Array.from({ length: FOREST_SETTINGS.trees }, () => {
    const sample = Array.from({ length: rows.length }, () => rows[Math.floor(rng() * rows.length)]);
    return tree(sample, 0, rng, dimensions);
  });
  return { trees, predict(x) {
    if (!x || x.length !== dimensions || !x.every(finite)) return null;
    return mean(trees.map(root => {
      let node = root;
      while (node.feature !== undefined) node = x[node.feature] < node.threshold ? node.left : node.right;
      return node.value;
    }));
  } };
}
export function fitTarget(pairs, key, cutoff, current, seed = 2026) {
  const selected = pairs.filter(pair => pair.day < cutoff && finite(pair.observed[key]) && finite(pair.raw[key]) && features(pair.raw));
  if (!current || !finite(current[key]) || !features(current)) return null;
  const forest = trainForest(selected.map(pair => ({ x: features(pair.raw), y: pair.observed[key] - pair.raw[key] })), seed);
  if (!forest) return null;
  const corrected = current[key] + forest.predict(features(current));
  return { forest, n: selected.length, raw: current[key], corrected: ['rain', 'gust'].includes(key) ? Math.max(0, corrected) : corrected };
}
export function validateTarget(pairs, key, cutoff, seed = 2026) {
  const train = pairs.filter(pair => pair.day < cutoff && finite(pair.observed[key]) && finite(pair.raw[key]) && features(pair.raw));
  const test = pairs.filter(pair => pair.day >= cutoff && finite(pair.observed[key]) && finite(pair.raw[key]) && features(pair.raw));
  const forest = trainForest(train.map(pair => ({ x: features(pair.raw), y: pair.observed[key] - pair.raw[key] })), seed);
  if (!forest) return { train: train.length, test: test.length, raw: null, corrected: null };
  const raw = test.map(pair => Math.abs(pair.raw[key] - pair.observed[key]));
  const corrected = test.map(pair => {
    const value = pair.raw[key] + forest.predict(features(pair.raw));
    return Math.abs((['rain', 'gust'].includes(key) ? Math.max(0, value) : value) - pair.observed[key]);
  });
  return { train: train.length, test: test.length,
    raw: test.length >= 5 ? mean(raw) : null,
    corrected: test.length >= 5 ? mean(corrected) : null };
}

export const MODEL_LIFETIME_MS = 30 * 86400000;
export function trainBundle(pairs, target, current, trainedAt = Date.now()) {
  const recent = pairs.filter(pair => pair.day >= shiftDay(target, -120));
  const models = {}, validation = {};
  for (const [i, variable] of TARGETS.entries()) {
    const fit = fitTarget(recent, variable.key, target, current, 2026 + i);
    if (!fit) return null; // A saved station model must serve all four targets.
    models[variable.key] = { trees: fit.forest.trees, n: fit.n };
    validation[variable.key] = validateTarget(pairs, variable.key, shiftDay(target, -14), 2026 + i);
  }
  return {
    trainedAt, windowStart: shiftDay(target, -134), windowEnd: shiftDay(target, -1),
    models, validation
  };
}
export function reusableBundle(bundle, now = Date.now()) {
  return !!bundle && Number.isFinite(bundle.trainedAt) && now >= bundle.trainedAt &&
    now - bundle.trainedAt < MODEL_LIFETIME_MS &&
    TARGETS.every(variable => Array.isArray(bundle.models?.[variable.key]?.trees) &&
      bundle.models[variable.key].trees.length === FOREST_SETTINGS.trees);
}
export function predictForest(trees, x) {
  if (!Array.isArray(trees) || !trees.length || !x?.every(finite)) return null;
  return mean(trees.map(root => {
    let node = root;
    while (node.feature !== undefined) node = x[node.feature] < node.threshold ? node.left : node.right;
    return node.value;
  }));
}
export function predictBundle(bundle, current) {
  if (!reusableBundle(bundle) || !features(current)) return null;
  const x = features(current);
  const predictions = {};
  for (const variable of TARGETS) {
    const estimate = current[variable.key] + predictForest(bundle.models[variable.key].trees, x);
    predictions[variable.key] = {
      n: bundle.models[variable.key].n,
      corrected: ['rain', 'gust'].includes(variable.key) ? Math.max(0, estimate) : estimate
    };
  }
  if (predictions.low.corrected > predictions.high.corrected) {
    const midpoint = (predictions.low.corrected + predictions.high.corrected) / 2;
    predictions.low.corrected = predictions.high.corrected = midpoint;
    predictions.temperatureAdjusted = true;
  }
  return predictions;
}
