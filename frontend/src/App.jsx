import { useState } from "react"
import "./App.css"

const initialForm = {
  city: "São Paulo",
  area_m2: 85,
  bedrooms: 3,
  bathrooms: 2,
  floor: 8,
  parking_spaces: 1,
  neighborhood_score: 8.2,
  condo_fee: 650.0,
  age_years: 7,
  distance_to_center_km: 5.4,
}

const cities = [
  "São Paulo",
  "Rio de Janeiro",
  "Brasília",
  "Florianópolis",
  "Belo Horizonte",
  "Curitiba",
  "Porto Alegre",
  "Salvador",
  "Recife",
  "Fortaleza",
]

function App() {
  const [form, setForm] = useState(initialForm)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [showModelInfo, setShowModelInfo] = useState(false)

  async function handlePredict() {
    setLoading(true)
    setError("")
    setPrediction(null)

    try {
      const response = await fetch("/invocations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataframe_records: [form],
        }),
      })

      if (!response.ok) {
        throw new Error(await response.text())
      }

      const data = await response.json()
      setPrediction(data.predictions[0])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleLoadModelInfo() {
    setModelLoading(true)
    setError("")

    try {
      const response = await fetch("/model")

      if (!response.ok) {
        throw new Error(await response.text())
      }

      const data = await response.json()
      setModelInfo(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setModelLoading(false)
    }
  }

  function updateField(field, value) {
    setForm((prev) => ({
      ...prev,
      [field]: Number(value),
    }))
  }

  function updateTextField(field, value) {
    setForm((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  return (
    <main className="page">
      <section className="shell">
        <div className="header">
          <span className="badge">MLflow + FastAPI + Docker + MinIO</span>
          <h1>Previsão de Preço de Apartamento</h1>
          <p>
            Interface para consultar o modelo de regressão servido no backend FastAPI,
            com artefatos versionados no MLflow Registry e armazenados no MinIO.
          </p>
        </div>

        <div className="grid">
          <section className="card">
            <h2>Dados do imóvel</h2>

            <div className="form-grid">
              <CityField
                label="Cidade"
                value={form.city}
                options={cities}
                onChange={(v) => updateTextField("city", v)}
              />

              <Field label="Área (m²)" value={form.area_m2} onChange={(v) => updateField("area_m2", v)} />
              <Field label="Quartos" value={form.bedrooms} onChange={(v) => updateField("bedrooms", v)} />
              <Field label="Banheiros" value={form.bathrooms} onChange={(v) => updateField("bathrooms", v)} />
              <Field label="Andar" value={form.floor} onChange={(v) => updateField("floor", v)} />
              <Field label="Vagas" value={form.parking_spaces} onChange={(v) => updateField("parking_spaces", v)} />
              <Field label="Nota do bairro" value={form.neighborhood_score} onChange={(v) => updateField("neighborhood_score", v)} />
              <Field label="Condomínio" value={form.condo_fee} onChange={(v) => updateField("condo_fee", v)} />
              <Field label="Idade do imóvel" value={form.age_years} onChange={(v) => updateField("age_years", v)} />
              <Field label="Distância ao centro (km)" value={form.distance_to_center_km} onChange={(v) => updateField("distance_to_center_km", v)} />
            </div>

            <button className="primary-button" onClick={handlePredict} disabled={loading}>
              {loading ? "Calculando..." : "Gerar previsão"}
            </button>

            {error && <pre className="error">{error}</pre>}
          </section>

          <section className="right-column">
            <section className="result-card">
              <div className="collapsible-header">
                <div>
                  <span className="status">MLflow Registry</span>
                  <h2>Modelo ativo</h2>
                </div>

                <button
                  className="collapse-button"
                  onClick={() => setShowModelInfo((prev) => !prev)}
                  type="button"
                >
                  {showModelInfo ? "Recolher" : "Expandir"}
                </button>
              </div>

              {showModelInfo && (
                <>
                  <button className="secondary-button" onClick={handleLoadModelInfo} disabled={modelLoading}>
                    {modelLoading ? "Consultando modelo..." : "Carregar modelo ativo"}
                  </button>

                  {modelInfo ? (
                    <div className="details model-details">
                      <p><strong>Status:</strong> {modelInfo.status}</p>
                      <p><strong>Model URI:</strong> {modelInfo.model_uri}</p>
                      <p><strong>Model name:</strong> {modelInfo.model_name}</p>
                      <p><strong>Model version:</strong> {modelInfo.model_version ?? "N/A"}</p>
                      <p><strong>Model alias:</strong> {modelInfo.model_alias ?? "N/A"}</p>
                      <p><strong>Model source:</strong> {modelInfo.model_source ?? "N/A"}</p>
                      <p><strong>Tracking URI:</strong> {modelInfo.tracking_uri}</p>
                    </div>
                  ) : (
                    <div className="details model-details">
                      <p>Clique em <strong>Carregar modelo ativo</strong> para consultar o backend.</p>
                    </div>
                  )}
                </>
              )}
            </section>

            <section className="result-card">
              <span className="status">Backend conectado</span>
              <h2>Resultado</h2>

              <div className="price">
                {prediction === null
                  ? "Aguardando previsão"
                  : new Intl.NumberFormat("pt-BR", {
                      style: "currency",
                      currency: "BRL",
                    }).format(prediction)}
              </div>

              <div className="details">
                <p><strong>Cidade:</strong> {form.city}</p>
                <p><strong>Endpoint:</strong> /invocations</p>
                <p><strong>Modelo:</strong> {modelInfo?.model_name ?? "apartment-price-regression"}</p>
                <p><strong>Versão:</strong> {modelInfo?.model_version ?? "champion"}</p>
                <p><strong>URI:</strong> {modelInfo?.model_uri ?? "models:/apartment-price-regression@champion"}</p>
                <p><strong>Runtime:</strong> FastAPI + MLflow PyFunc</p>
              </div>
            </section>
          </section>
        </div>
      </section>
    </main>
  )
}

function Field({ label, value, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" step="any" value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}

function CityField({ label, value, options, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((city) => (
          <option key={city} value={city}>
            {city}
          </option>
        ))}
      </select>
    </label>
  )
}

export default App