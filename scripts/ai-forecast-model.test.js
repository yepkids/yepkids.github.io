// In the local preview console: (await import('/scripts/ai-forecast-model.test.js')).runTests()
import { FOREST_SETTINGS, shiftDay, cycleBounds, aggregateForecast, parseAsosCSV,
  aggregateObservations, features, trainForest, fitTarget, validateTarget,
  MODEL_LIFETIME_MS, trainBundle, reusableBundle, predictBundle } from '../assets/js/ai-forecast-model.js';

export function runTests() {
  let passed = 0;
  function check(condition, message) {
    if (!condition) throw new Error(message);
    passed++;
  }
  function hourly(day) {
    const start = cycleBounds(day).start - 43200;
    const time = Array.from({ length: 72 }, (_, i) => start + i * 3600);
    const values = value => time.map(() => value);
    return { time, temperature_2m: values(50), precipitation: values(1),
      wind_gusts_10m: values(20), relative_humidity_2m: values(55),
      cloud_cover: values(40), cape: values(100), wind_direction_10m: values(359),
      wind_speed_10m: values(15) };
  }
  check(shiftDay('2026-01-01', -1) === '2025-12-31', 'Date arithmetic');
  check(cycleBounds('2026-10-28').end - cycleBounds('2026-10-28').start === 86400, 'Fixed UTC verification window');
  for (const day of ['2026-03-08', '2026-11-01']) {
    const result = aggregateForecast(hourly(day), day);
    check(result.rain === 24 && result.low === 50 && result.high === 50, 'UTC window ignores local daylight saving');
    check(result.rhMin === 55 && result.rhMax === 55 && result.cloudMean === 40 && result.capeMean === 100 && result.windMax === 15, 'Additional inputs aggregate correctly');
    check(result.direction > 358 && result.direction < 360, 'Wind direction is circular');
  }
  const boundary = hourly('2026-10-28');
  boundary.precipitation.fill(0);
  const firstEnd = cycleBounds('2026-10-28').start + 3600;
  boundary.precipitation[boundary.time.indexOf(firstEnd)] = 2;
  boundary.precipitation[boundary.time.indexOf(cycleBounds('2026-10-28').start)] = 100;
  check(aggregateForecast(boundary, '2026-10-28').rain === 2, 'Precipitation uses ending hour');
  const incomplete = hourly('2026-10-28');
  incomplete.cape[incomplete.time.indexOf(firstEnd)] = null;
  check(aggregateForecast(incomplete, '2026-10-28').capeMean === null, 'Missing predictor invalidates complete feature vector');
  check(features(aggregateForecast(incomplete, '2026-10-28')) === null, 'Incomplete feature vector excluded');
  const complete = aggregateForecast(hourly('2026-10-28'), '2026-10-28');
  check(features(complete).length === 11, 'Four raw forecasts plus six added inputs with circular direction');
  const csv = 'station,valid,tmpf,gust,sknt,p01i\nMSP,2026-10-28 06:53,50.00,M,8.00,T\nMSP,2026-10-28 07:53,51.00,22.00,11.00,0.01\n';
  const parsed = parseAsosCSV(csv);
  check(parsed.length === 2 && parsed[0].p01i === 0 && parsed[0].gust === null, 'METAR CSV parses trace and missing values');
  check(aggregateObservations(parsed, '2026-10-28').low === null, 'Sparse METAR cycles excluded');
  const observations = Array.from({ length: 24 }, (_, i) => ({
    time: cycleBounds('2026-10-28').start + i * 3600 + 3180, tmpf: 50 + i,
    p01i: 0.01, sknt: 10, gust: i === 12 ? 25 : null
  }));
  const obs = aggregateObservations(observations, '2026-10-28');
  check(obs.low === 50 && obs.high === 73 && obs.gust === 25, 'METAR cycle temperature and gust extrema');
  check(Math.abs(obs.rain - 0.24) < 1e-9, 'METAR hourly precipitation sum');
  check(aggregateObservations(observations.map(row => ({ ...row, gust: null })), '2026-10-28').gust === 0, 'No reported gust gives zero');
  const pairs = Array.from({ length: 134 }, (_, i) => {
    const day = shiftDay('2026-06-16', i);
    const raw = { ...complete, low: 40 + i % 10, high: 60 + i % 10 };
    return { day, raw, observed: { low: raw.low + (i < 120 ? 2 : 20), high: raw.high + 2, rain: 25, gust: 22 } };
  });
  const cutoff = pairs[120].day;
  const training = pairs.slice(0, 120).map(pair => ({ x: features(pair.raw), y: 2 }));
  const forest = trainForest(training, 17);
  check(forest.predict(features(complete)) === 2, 'Constant residual forest');
  const second = trainForest(training, 17);
  check(second.predict(features(complete)) === forest.predict(features(complete)), 'Seeded forest reproducible');
  const depth = node => node.feature === undefined ? 0 : 1 + Math.max(depth(node.left), depth(node.right));
  check(forest.trees.every(node => depth(node) <= 4), 'Maximum tree depth');
  check(FOREST_SETTINGS.maxDepth === 4 && FOREST_SETTINGS.minSamplesLeaf === 5 && FOREST_SETTINGS.maxFeatures === 'sqrt', 'Requested hyperparameters');
  const holdout = validateTarget(pairs, 'low', cutoff);
  check(holdout.train === 120 && holdout.test === 14, 'Chronological 120/14 split');
  check(holdout.raw > holdout.corrected && holdout.corrected > 17, 'Holdout does not leak into fitted forest');
  const fitted = fitTarget(pairs.slice(-120), 'low', shiftDay(pairs.at(-1).day, 1), complete);
  check(fitted.n === 120, 'Final fit uses all completed cycles');
  check(fitTarget(pairs.slice(0, 29), 'low', cutoff, complete) === null, 'At least 30 training cycles required');
  check(fitTarget(pairs, 'low', cutoff, { ...complete, capeMean: null }) === null, 'Missing current input rejects forecast');
  const short = validateTarget(pairs.slice(0, 124), 'low', cutoff);
  check(short.test === 4 && short.raw === null && short.corrected === null, 'At least five holdout cycles required');
  const now = Date.now();
  const bundle = trainBundle(pairs, shiftDay(pairs.at(-1).day, 1), complete, now);
  check(reusableBundle(bundle, now + MODEL_LIFETIME_MS - 1), 'Station model reusable through day 30');
  check(!reusableBundle(bundle, now + MODEL_LIFETIME_MS), 'Station model expires after 30 days');
  const restored = JSON.parse(JSON.stringify(bundle));
  check(reusableBundle(restored, now), 'Station forest survives browser-storage serialization');
  const firstForecast = predictBundle(restored, complete);
  const nextRun = { ...complete, low: complete.low + 10 };
  const nextForecast = predictBundle(restored, nextRun);
  check(firstForecast.low.n === 120 && Number.isFinite(nextForecast.low.corrected), 'Saved forest has all training samples');
  check(nextForecast.low.corrected !== firstForecast.low.corrected, 'New ECMWF forecast uses saved forest without retraining');
  return `${passed} assertions passed`;
}
