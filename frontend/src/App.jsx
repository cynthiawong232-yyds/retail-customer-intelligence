import { useEffect, useState } from "react";

// Set in .env.local for development, and in Vercel's project settings for
// production. Falls back to localhost so `npm run dev` works with no setup.
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${res.status} on ${path}`);
  return res.json();
}

const gbp = (n) =>
  new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(n);

/* ------------------------------------------------------------------ */

function Panel({ title, subtitle, children }) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
        {subtitle && <p className="sub">{subtitle}</p>}
      </header>
      {children}
    </section>
  );
}

function Caveat({ children }) {
  // Every model's limitation is shown next to its number, not in a footnote.
  // A prediction without its caveat is how an uncalibrated probability ends
  // up multiplied by a budget in somebody's spreadsheet.
  return <p className="caveat">{children}</p>;
}

/* ------------------------------------------------------------------ */

function Segment({ data }) {
  if (!data) return <p className="muted">loading</p>;
  return (
    <>
      <p className="headline">{data.segment_name}</p>
      <p className="sub">{data.interpretation}</p>
      <dl className="kv">
        <dt>recency</dt>
        <dd>{data.inputs.recency} days</dd>
        <dt>frequency</dt>
        <dd>{data.inputs.frequency} orders</dd>
        <dt>monetary</dt>
        <dd>{gbp(data.inputs.monetary)}</dd>
        <dt>distance to centroid</dt>
        <dd>{data.distance_to_centroid}</dd>
      </dl>
      <Caveat>
        A segment describes a group, not a person. Distance to centroid is
        shown because a customer far from every centroid is one the model is
        not really sure about.
      </Caveat>
    </>
  );
}

function Clv({ data }) {
  if (!data) return <p className="muted">loading</p>;
  const alive = data.p_still_alive;
  return (
    <>
      <p className="headline">{gbp(data.expected_spend_90d)}</p>
      <p className="sub">expected spend, next 90 days</p>

      <div className="bar" title={`P(still alive) = ${alive}`}>
        <div className="bar-fill" style={{ width: `${alive * 100}%` }} />
      </div>
      <p className="sub">
        {(alive * 100).toFixed(1)}% chance still an active customer &middot;
        value decile {data.value_decile} of 10
      </p>

      <dl className="kv">
        <dt>challenger ({data.challenger.model})</dt>
        <dd>{gbp(data.challenger.estimate)}</dd>
        <dt>P(buys at all)</dt>
        <dd>{(data.challenger.p_buys_at_all * 100).toFixed(1)}%</dd>
        <dt>if they buy</dt>
        <dd>{gbp(data.challenger.expected_amount_if_they_buy)}</dd>
      </dl>
      <Caveat>{data.caveat}</Caveat>
    </>
  );
}

function Repurchase({ data }) {
  if (!data) return <p className="muted">loading</p>;
  const pct = data.probability * 100;
  return (
    <>
      <p className="headline">{pct.toFixed(0)}%</p>
      <p className="sub">
        chance of buying within 90 days &middot; decile {data.decile} of 10
      </p>

      <div className="bar">
        <div className="bar-fill" style={{ width: `${pct}%` }} />
      </div>

      <h3>What moved this prediction</h3>
      <ul className="drivers">
        {data.top_drivers.map((d) => (
          <li key={d.feature}>
            <span className={d.direction === "raises" ? "up" : "down"}>
              {d.direction === "raises" ? "▲" : "▼"}
            </span>
            <span className="feat">{d.feature}</span>
            <span className="val">{d.value ?? "missing"}</span>
            <span className="eff">{d.effect_log_odds.toFixed(3)}</span>
          </li>
        ))}
      </ul>
      <p className="sub">
        SHAP contributions in log-odds. Precomputed, so the container never
        imports the shap library.
      </p>
      <Caveat>{data.caveat}</Caveat>
    </>
  );
}

function Recommendations({ data, onPick }) {
  if (!data) return <p className="muted">loading</p>;
  return (
    <>
      <p className="sub">{data.reading}</p>
      <ol className="recs">
        {data.recommendations.map((r) => (
          <li key={r.stock_code}>
            <button className="link" onClick={() => onPick(r.stock_code)}>
              {r.description || r.stock_code}
            </button>
            <span className={r.bought_before ? "tag reorder" : "tag new"}>
              {r.bought_before ? "reorder" : "new"}
            </span>
          </li>
        ))}
      </ol>
      <Caveat>{data.caveat}</Caveat>
    </>
  );
}

function Similar({ code, data, loading }) {
  return (
    <>
      <p className="sub">
        Nearest neighbours in 64-dimensional item2vec space, computed live.
        Nobody told the model what these products are. It only saw which ones
        land in the same basket.
      </p>
      {!code && <p className="muted">click a recommended product</p>}
      {loading && <p className="muted">loading</p>}
      {data && (
        <ul className="similar">
          {data.map((s) => (
            <li key={s.stock_code}>
              <span className="sim">{s.similarity.toFixed(3)}</span>
              <span>{s.description || s.stock_code}</span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */

export default function App() {
  const [health, setHealth] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [panels, setPanels] = useState({});
  const [code, setCode] = useState("22423");
  const [similar, setSimilar] = useState(null);
  const [loadingSim, setLoadingSim] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([get("/health"), get("/customers?limit=60")])
      .then(([h, c]) => {
        setHealth(h);
        setCustomers(c.customers);
        setSelected(c.customers[0]?.customer_id ?? null);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (selected == null) return;
    setPanels({});
    // Four independent endpoints, fetched in parallel. One failing must not
    // blank the other three, so each settles on its own.
    const jobs = {
      segment: `/customers/${selected}`,
      clv: `/predict/clv/${selected}`,
      repurchase: `/predict/repurchase/${selected}`,
      recommend: `/recommend/${selected}`,
    };
    Object.entries(jobs).forEach(([key, path]) => {
      get(path)
        .then((d) => setPanels((p) => ({ ...p, [key]: d })))
        .catch(() => setPanels((p) => ({ ...p, [key]: null })));
    });
  }, [selected]);

  useEffect(() => {
    if (!code) return;
    setLoadingSim(true);
    get(`/similar/${encodeURIComponent(code)}?k=6`)
      .then(setSimilar)
      .catch(() => setSimilar([]))
      .finally(() => setLoadingSim(false));
  }, [code]);

  if (error) {
    return (
      <main className="wrap">
        <h1>Retail Customer Intelligence</h1>
        <p className="error">
          Cannot reach the API at <code>{API}</code>. {error}
        </p>
        <p className="sub">
          Start it with <code>uvicorn rci.api:app --reload</code>, or set
          <code> VITE_API_URL</code>.
        </p>
      </main>
    );
  }

  return (
    <main className="wrap">
      <h1>Retail Customer Intelligence</h1>
      <p className="sub lede">
        Four models on 776,577 real UK retail transactions, split by time and
        served from one API. Customer data as of{" "}
        {health?.customer_features_as_of ?? "..."}.
      </p>

      <div className="picker">
        <label htmlFor="cust">Customer</label>
        <select
          id="cust"
          value={selected ?? ""}
          onChange={(e) => setSelected(Number(e.target.value))}
        >
          {customers.map((c) => (
            <option key={c.customer_id} value={c.customer_id}>
              {c.customer_id} &middot; {c.frequency} orders &middot;{" "}
              {gbp(c.monetary)} &middot; {c.recency}d ago
            </option>
          ))}
        </select>
      </div>

      <div className="grid">
        <Panel title="Who are they?" subtitle="KMeans, k=4, on log-scaled RFM">
          <Segment data={panels.segment} />
        </Panel>

        <Panel
          title="What are they worth?"
          subtitle="BG/NBD + Gamma-Gamma, with an XGBoost challenger"
        >
          <Clv data={panels.clv} />
        </Panel>

        <Panel
          title="Will they come back?"
          subtitle="XGBoost, 35 trees, explained with SHAP"
        >
          <Repurchase data={panels.repurchase} />
        </Panel>

        <Panel
          title="What do we show them?"
          subtitle="item2vec retrieval, then an XGBoost ranker"
        >
          <Recommendations data={panels.recommend} onPick={setCode} />
        </Panel>

        <Panel title={`Products similar to ${code}`} subtitle="live, no lookup">
          <Similar code={code} data={similar} loading={loadingSim} />
        </Panel>
      </div>

      <footer>
        <p className="sub">
          Data: UCI Online Retail II, CC BY 4.0. No synthetic data. Trained on
          the {health?.model_trained_on_cutoff ?? "..."} snapshot and scored on
          a later one, which is why the probabilities carry a calibration
          caveat.
        </p>
      </footer>
    </main>
  );
}
