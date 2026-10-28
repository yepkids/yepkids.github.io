import { TARGETS, INPUTS, HOURLY_FIELDS, finite, shiftDay, cycleBounds,
  aggregateForecast, parseAsosCSV, aggregateObservations, features,
  MODEL_LIFETIME_MS, trainBundle, reusableBundle, predictBundle } from './ai-forecast-model.js';

const root = document.getElementById('ai-forecast');
const el = id => document.getElementById(`af-${id}`);
const MODEL = { id: 'ecmwf_ifs', name: 'ECMWF IFS HRES 9 km' };
const MODEL_CACHE_VERSION = 'ecmwf12-rf-v1';
const STATES = 'AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR VI GU AS'.split(' ');
let stations = [], controller;
const memoryCache = new Map();
let databasePromise;
function database() {
  if (!databasePromise) databasePromise = new Promise(resolve => {
    if (!globalThis.indexedDB) return resolve(null);
    let request;
    try { request = indexedDB.open('ai-forecast-v2', 2); } catch { resolve(null); return; }
    request.onupgradeneeded = () => {
      for (const name of ['runs', 'models']) {
        if (!request.result.objectStoreNames.contains(name)) request.result.createObjectStore(name);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
  return databasePromise;
}
async function cached(storeName, key, value) {
  const db = await database();
  if (!db) {
    if (storeName === 'models') {
      try {
        if (value === undefined) return JSON.parse(localStorage.getItem(key));
        localStorage.setItem(key, JSON.stringify(value));
        return value;
      } catch { return undefined; }
    }
    return value === undefined ? memoryCache.get(key) : memoryCache.set(key, value);
  }
  return new Promise(resolve => {
    try {
      const transaction = db.transaction(storeName, value === undefined ? 'readonly' : 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = value === undefined ? store.get(key) : store.put(value, key);
      if (value === undefined) request.onsuccess = () => resolve(request.result);
      else transaction.oncomplete = () => resolve(value);
      request.onerror = () => resolve(undefined);
      transaction.onabort = () => resolve(undefined);
    } catch { resolve(undefined); }
  });
}
function modelCacheKey(station) {
  return `ai-forecast-model|${MODEL_CACHE_VERSION}|${station.properties.network}|${station.id}|${station.geometry.coordinates.join(',')}`;
}
async function request(url, signal, asText = false) {
  const timeout = new AbortController();
  const abort = () => timeout.abort(signal?.reason);
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  const timer = setTimeout(() => timeout.abort(), 60000);
  try {
    const response = await fetch(url, { signal: timeout.signal });
    const body = asText ? await response.text() : await response.json();
    if (!response.ok || body?.error) throw new Error(`HTTP ${response.status}: ${body?.reason || 'Data service request failed'}`);
    return body;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
const option = (value, text) => new Option(text, value);
const stationCode = id => /^[A-Z0-9]{3}$/.test(id) ? `K${id}` : id;
async function loadStations() {
  el('generate').disabled = true;
  el('station').replaceChildren(option('', 'Loading stations…'));
  el('status').textContent = 'Loading station directory…';
  const state = el('state').value;
  try {
    const data = await request(`https://mesonet.agron.iastate.edu/geojson/network/${state}_ASOS.geojson`);
    if (state !== el('state').value) return;
    stations = data.features.filter(feature => feature.properties.online && feature.geometry?.coordinates).sort((a, b) => a.properties.sname.localeCompare(b.properties.sname));
    el('station').replaceChildren(...stations.map(feature => option(feature.id, `${stationCode(feature.id)} — ${feature.properties.sname}`)));
    const defaultID = root.dataset.defaultStation.replace(/^K(?=[A-Z0-9]{3}$)/, '');
    if (state === root.dataset.defaultState && stations.some(feature => feature.id === defaultID)) el('station').value = defaultID;
    el('generate').disabled = !stations.length;
    el('status').textContent = stations.length ? 'Choose a station and generate the next ECMWF 12 UTC forecast.' : 'No active stations available in this directory.';
  } catch (error) {
    if (state !== el('state').value) return;
    el('station').replaceChildren(option('', 'Station directory unavailable'));
    el('status').textContent = `Could not load stations: ${error.message}. Change state to retry.`;
  }
}
async function modelRun(station, runDay, targetDay, signal, historical) {
  const [longitude, latitude] = station.geometry.coordinates;
  const url = new URL('https://single-runs-api.open-meteo.com/v1/forecast');
  url.search = new URLSearchParams({ latitude, longitude, run: `${runDay}T12:00`, models: MODEL.id,
    hourly: HOURLY_FIELDS, temperature_unit: 'fahrenheit', wind_speed_unit: 'kn', precipitation_unit: 'inch', timeformat: 'unixtime' });
  const key = `${url}|${targetDay}`;
  const saved = historical ? await cached('runs', key) : null;
  if (saved && Date.now() - saved.created < 7 * 86400000) return saved.values;
  const data = await request(url, signal);
  const values = aggregateForecast(data.hourly, targetDay);
  if (!features(values)) throw new Error('A required forecast input is missing from this run');
  if (historical) await cached('runs', key, { created: Date.now(), values });
  return values;
}
async function latestRun(station, signal) {
  const now = new Date();
  const today = now.toISOString().slice(0, 10);
  const candidates = now.getUTCHours() >= 12 ? [today, shiftDay(today, -1)] : [shiftDay(today, -1)];
  for (const [i, runDay] of candidates.entries()) {
    signal.throwIfAborted();
    el('status').textContent = `Checking ECMWF ${runDay} 12 UTC…`;
    try {
      const target = shiftDay(runDay, 1);
      return { runDay, target, current: await modelRun(station, runDay, target, signal, false) };
    } catch (error) {
      signal.throwIfAborted();
      if (i === candidates.length - 1 || !/requested model run is not available/i.test(error.message)) throw error;
    }
  }
  throw new Error('No ECMWF 12 UTC run is available.');
}
async function observations(station, days, signal) {
  const first = days[0], last = days.at(-1);
  const months = [];
  for (let month = first.slice(0, 7); month <= shiftDay(last, 1).slice(0, 7); month = shiftDay(`${month}-01`, 32).slice(0, 7)) months.push(month);
  const rows = [];
  for (const month of months) {
    signal.throwIfAborted();
    const next = shiftDay(`${month}-01`, 32).slice(0, 7);
    const url = new URL('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py');
    url.search = new URLSearchParams({ station: station.id, data: 'tmpf,gust,sknt,p01i',
      year1: month.slice(0, 4), month1: Number(month.slice(5)), day1: 1,
      year2: next.slice(0, 4), month2: Number(next.slice(5)), day2: 1,
      tz: 'Etc/UTC', format: 'onlycomma', latlon: 'no', elev: 'no',
      missing: 'M', trace: 'T', direct: 'no', report_type: '3,4' });
    // URLSearchParams must repeat parameters for the IEM ASOS request.
    url.searchParams.delete('data');
    for (const field of ['tmpf', 'gust', 'sknt', 'p01i']) url.searchParams.append('data', field);
    url.searchParams.delete('report_type');
    url.searchParams.append('report_type', '3');
    url.searchParams.append('report_type', '4');
    el('status').textContent = `Downloading station reports for ${month}…`;
    rows.push(...parseAsosCSV(await request(url, signal, true)));
  }
  return new Map(days.map(day => [day, aggregateObservations(rows, day)]));
}
function cell(text, tag = 'td') { const node = document.createElement(tag); node.textContent = text; return node; }
const format = (value, variable) => finite(value) ? value.toFixed(variable.digits) : 'Unavailable';
function render(station, runDay, target, bundle, current, warnings, reused) {
  const { start, end } = cycleBounds(target);
  el('heading').textContent = `${stationCode(station.id)} — ${station.properties.sname}`;
  el('period').textContent = `${new Date(start * 1000).toISOString()} to ${new Date(end * 1000).toISOString()}; ECMWF initialized ${runDay} at 12 UTC. Generated ${new Date().toISOString()}.`;
  el('cards').replaceChildren(); el('models').replaceChildren(); el('validation').replaceChildren(); el('inputs').replaceChildren();
  const predictions = predictBundle(bundle, current);
  if (predictions.temperatureAdjusted) {
    warnings.push('Predicted minimum exceeded maximum; both temperatures were adjusted to their midpoint.');
  }
  for (const variable of TARGETS) {
    const fit = predictions[variable.key];
    const card = document.createElement('div'); card.className = 'af-card';
    card.append(cell(variable.label, 'span'), cell(`${format(fit.corrected, variable)} ${variable.unit}`, 'strong'), cell(`${fit.n} matched days`, 'span'));
    el('cards').append(card);
    const validation = bundle.validation[variable.key];
    const row = document.createElement('tr');
    row.append(cell(`${variable.label} (${variable.unit})`, 'th'), cell(validation.train), cell(validation.test), cell(format(validation.raw, variable)), cell(format(validation.corrected, variable)));
    el('validation').append(row);
  }
  for (const [label, values] of [['ECMWF raw', current], ['Random forest forecast', Object.fromEntries(TARGETS.map(variable => [variable.key, predictions[variable.key].corrected]))]]) {
    const row = document.createElement('tr'); row.append(cell(label, 'th'));
    for (const variable of TARGETS) row.append(cell(format(values[variable.key], variable)));
    el('models').append(row);
  }
  const modelInputs = [
    ...TARGETS.map(variable => ({
      ...variable,
      label: variable.key === 'rain' ? 'Precipitation sum' :
        variable.key === 'gust' ? 'Maximum wind gust' : variable.label
    })),
    ...INPUTS
  ];
  for (const variable of modelInputs) {
    const row = document.createElement('tr');
    row.append(cell(variable.label, 'th'), cell(`${format(current[variable.key], variable)} ${variable.unit}`));
    el('inputs').append(row);
  }
  el('training').textContent = `Station model trained ${new Date(bundle.trainedAt).toISOString()} using ${bundle.windowStart} through ${bundle.windowEnd} (134 historical 06–06 UTC cycles). Backtest: 120 earlier training cycles, followed by 14 unseen test cycles. The displayed forecast uses the latest 120 cycles available at training time. ${reused ? 'This forecast reused the saved station model' : 'This forecast trained a new station model'}; it expires ${new Date(bundle.trainedAt + MODEL_LIFETIME_MS).toISOString()}. The current ECMWF run is fetched on every generation. Counts vary by target; temperature ordering adjustments are outside target-wise MAE.`;
  el('warnings').replaceChildren(...warnings.map(message => cell(message, 'li')));
  el('results').hidden = false;
  return true;
}
el('form').addEventListener('submit', async event => {
  event.preventDefault();
  const station = stations.find(item => item.id === el('station').value);
  if (!station) return;
  controller = new AbortController();
  const { signal } = controller;
  el('results').hidden = true;
  el('generate').disabled = el('state').disabled = el('station').disabled = true;
  el('cancel').hidden = false; el('progress').hidden = false; el('progress').value = 0;
  try {
    const { runDay, target, current } = await latestRun(station, signal);
    const days = Array.from({ length: 134 }, (_, i) => shiftDay(target, i - 134));
    const key = modelCacheKey(station);
    const saved = await cached('models', key);
    const metarNote = 'METAR reports are usually hourly: observed extremes and precipitation near 06 UTC may differ from official WxChallenge verification. Trace precipitation is treated as zero.';
    if (reusableBundle(saved)) {
      signal.throwIfAborted();
      render(station, runDay, target, saved, current, [metarNote], true);
      el('status').textContent = 'Forecast ready using the saved station model and latest ECMWF 12 UTC run.';
      return;
    }
    const observed = await observations(station, days, signal);
    const pairs = [], warnings = [];
    for (const [i, day] of days.entries()) {
      signal.throwIfAborted();
      el('status').textContent = `Learning from ${day} 06 UTC (${i + 1}/134)…`;
      try {
        const raw = await modelRun(station, shiftDay(day, -1), day, signal, true);
        pairs.push({ day, raw, observed: observed.get(day) });
      } catch (error) {
        signal.throwIfAborted();
        warnings.push(`${day}: ${error.message}`);
        if (/limit|429/i.test(error.message)) throw new Error('Open-Meteo rate limit reached. Wait before retrying; completed downloads are cached.');
      }
      el('progress').value = (i + 1) / 134 * 100;
      await new Promise(resolve => setTimeout(resolve, 150));
    }
    signal.throwIfAborted();
    if (warnings.length > 6) warnings.splice(6, warnings.length - 6, `${warnings.length - 6} additional archived runs were unavailable.`);
    warnings.push(metarNote);
    const bundle = trainBundle(pairs, target, current);
    if (!bundle) {
      el('status').textContent = 'Not enough complete historical cycles to train all four forecasts.';
      return;
    }
    signal.throwIfAborted();
    if (!await cached('models', key, bundle)) warnings.push('Browser storage is unavailable; this station model may need training again next time.');
    render(station, runDay, target, bundle, current, warnings, false);
    el('status').textContent = 'Forecast ready. This station model is saved in this browser for 30 days.';
  } catch (error) {
    el('status').textContent = signal.aborted ? 'Forecast cancelled. Completed downloads remain cached.' : `Forecast could not be generated: ${error.message}`;
  } finally {
    el('generate').disabled = el('state').disabled = el('station').disabled = false;
    el('cancel').hidden = true; el('progress').hidden = true;
  }
});
el('cancel').addEventListener('click', () => controller?.abort());
el('state').replaceChildren(...STATES.map(state => option(state, state)));
el('state').value = root.dataset.defaultState || 'MN';
el('state').addEventListener('change', () => { el('results').hidden = true; loadStations(); });
el('station').addEventListener('change', () => { el('results').hidden = true; });
loadStations();
