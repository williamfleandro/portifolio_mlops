import { useEffect, useMemo, useState } from "react"
import "./App.css"

const API = {
  health: "/health",
  model: "/model",
  schema: "/schema",
  predict: "/predict",
}

const cityBasePriceM2 = {
  "Maceió": 9908,
  "Manaus": 7513,
  "Salvador": 8385,
  "Fortaleza": 9350,
  "Brasília": 10090,
  "Vitória": 14818,
  "Goiânia": 8226,
  "São Luís": 8627,
  "Cuiabá": 6931,
  "Campo Grande": 6839,
  "Belo Horizonte": 10663,
  "Belém": 8882,
  "João Pessoa": 8081,
  "Curitiba": 11694,
  "Recife": 8615,
  "Teresina": 5857,
  "Rio de Janeiro": 10939,
  "Natal": 6334,
  "Porto Alegre": 7579,
  "Florianópolis": 13208,
  "São Paulo": 12019,
  "Aracaju": 5529,
  "Rio Branco": 5200,
  "Macapá": 5400,
  "Porto Velho": 5600,
  "Boa Vista": 5300,
  "Palmas": 5900,
}

const cities = Object.keys(cityBasePriceM2)

const neighborhoods = [
  "Centro",
  "Área Nobre",
  "Zona Sul",
  "Zona Norte",
  "Zona Leste",
  "Zona Oeste",
  "Região Residencial",
  "Região Comercial",
  "Área Universitária",
  "Região Periférica",
]

const propertyTypes = [
  { value: "apartment", label: "Apartamento" },
  { value: "house", label: "Casa" },
]

const initialForm = {
  property_type: "apartment",
  city: "São Paulo",
  neighborhood: "Área Nobre",
  base_price_m2: cityBasePriceM2["São Paulo"],
  area_m2: 80,
  bedrooms: 2,
  bathrooms: 2,
  floor: 12,
  parking_spaces: 1,
  neighborhood_score: 9.2,
  condo_fee: 850,
  age_years: 5,
  distance_to_center_km: 4.5,
}

function formatBRL(value) {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(Number(value || 0))
}

function toNumber(value) {
  if (value === "" || value === null || value === undefined) {
    return 0
  }

  return Number(value)
}

function buildPayload(form) {
  return {
    dataframe_records: [
      {
        property_type: String(form.property_type || "apartment"),
        city: String(form.city || "São Paulo"),
        neighborhood: String(form.neighborhood || "Região Residencial"),
        base_price_m2: toNumber(form.base_price_m2),
        area_m2: toNumber(form.area_m2),
        bedrooms: toNumber(form.bedrooms),
        bathrooms: toNumber(form.bathrooms),
        floor: toNumber(form.floor),
        parking_spaces: toNumber(form.parking_spaces),
        neighborhood_score: toNumber(form.neighborhood_score),
        condo_fee: toNumber(form.condo_fee),
        age_years: toNumber(form.age_years),
        distance_to_center_km: toNumber(form.distance_to_center_km),
      },
    ],
  }
}

function validatePayload(payload) {
  const row = payload.dataframe_records[0]

  const requiredNumericFields = [
    "base_price_m2",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floor",
    "parking_spaces",
    "neighborhood_score",
    "condo_fee",
    "age_years",
    "distance_to_center_km",
  ]

  for (const field of requiredNumericFields) {
    if (!Number.isFinite(Number(row[field]))) {
      throw new Error(`Campo numérico inválido: ${field}`)
    }
  }

  if (!row.property_type) {
    throw new Error("Tipo do imóvel é obrigatório.")
  }

  if (!row.neighborhood) {
    throw new Error("Perfil do bairro é obrigatório.")
  }

  return payload
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, options)
  const text = await response.text()

  let data = null

  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = text
    }
  }

  if (!response.ok) {
    const detail =
      typeof data === "string"
        ? data
        : JSON.stringify(data, null, 2)

    throw new Error(`HTTP ${response.status} em ${url}\n${detail}`)
  }

  return data
}

function App() {
  const [form, setForm] = useState(initialForm)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [schemaInfo, setSchemaInfo] = useState(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [showModelInfo, setShowModelInfo] = useState(true)
  const [lastPayload, setLastPayload] = useState(null)

  const estimatedMarketValue = useMemo(() => {
    return toNumber(form.base_price_m2) * toNumber(form.area_m2)
  }, [form.base_price_m2, form.area_m2])

  useEffect(() => {
    handleLoadModelInfo({ silent: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handlePredict() {
    setLoading(true)
    setError("")
    setPrediction(null)

    try {
      const payload = validatePayload(buildPayload(form))
      setLastPayload(payload)

      const data = await requestJSON(API.predict, {
        method: "POST",
        headers: {
          "Content-Type": "application/json; charset=utf-8",
        },
        body: JSON.stringify(payload),
      })

      const predictedValue = Array.isArray(data.predictions)
        ? data.predictions[0]
        : null

      if (predictedValue === null || predictedValue === undefined) {
        throw new Error(`Resposta sem predictions: ${JSON.stringify(data, null, 2)}`)
      }

      setPrediction(predictedValue)

      setModelInfo((prev) => ({
        ...(prev || {}),
        status: "loaded",
        model_uri: data.model_uri || prev?.model_uri,
        model_name: data.model_name || prev?.model_name,
        model_version: data.model_version || prev?.model_version,
        model_alias: data.model_alias || prev?.model_alias,
      }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleLoadModelInfo(options = {}) {
    setModelLoading(true)

    if (!options.silent) {
      setError("")
    }

    try {
      const [modelData, schemaData] = await Promise.all([
        requestJSON(API.model),
        requestJSON(API.schema),
      ])

      setModelInfo(modelData)
      setSchemaInfo(schemaData)
    } catch (err) {
      if (!options.silent) {
        setError(err.message)
      }
    } finally {
      setModelLoading(false)
    }
  }

  function updateNumericField(field, value) {
    setForm((prev) => ({
      ...prev,
      [field]: value === "" ? "" : Number(value),
    }))
  }

  function updateTextField(field, value) {
    setForm((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  function updateCity(city) {
    setForm((prev) => ({
      ...prev,
      city,
      base_price_m2: cityBasePriceM2[city] ?? prev.base_price_m2,
    }))
  }

  function updatePropertyType(propertyType) {
    setForm((prev) => {
      if (propertyType === "house") {
        return {
          ...prev,
          property_type: propertyType,
          floor: 0,
          condo_fee: Math.min(toNumber(prev.condo_fee), 350),
          parking_spaces: Math.max(toNumber(prev.parking_spaces), 1),
        }
      }

      return {
        ...prev,
        property_type: propertyType,
      }
    })
  }

  return (
    <main className="page">
      <section className="shell">
        <div className="header">
          <span className="badge">MLflow + FastAPI + Docker + MinIO + Argo Rollouts</span>
          <h1>Previsão de Preço de Imóvel</h1>
          <p>
            Interface integrada ao backend FastAPI 1.4, usando o modelo
            <strong> candidate v9</strong> registrado no MLflow.
          </p>
        </div>

        <div className="grid">
          <section className="card">
            <h2>Dados do imóvel</h2>

            <div className="form-grid">
              <SelectField
                label="Tipo do imóvel"
                value={form.property_type}
                options={propertyTypes}
                onChange={updatePropertyType}
              />

              <SelectField
                label="Cidade"
                value={form.city}
                options={cities.map((city) => ({ value: city, label: city }))}
                onChange={updateCity}
              />

              <SelectField
                label="Perfil do bairro"
                value={form.neighborhood}
                options={neighborhoods.map((item) => ({ value: item, label: item }))}
                onChange={(value) => updateTextField("neighborhood", value)}
              />

              <Field
                label="Preço base por m²"
                value={form.base_price_m2}
                onChange={(value) => updateNumericField("base_price_m2", value)}
              />

              <Field
                label="Área (m²)"
                value={form.area_m2}
                onChange={(value) => updateNumericField("area_m2", value)}
              />

              <Field
                label="Quartos"
                value={form.bedrooms}
                onChange={(value) => updateNumericField("bedrooms", value)}
              />

              <Field
                label="Banheiros"
                value={form.bathrooms}
                onChange={(value) => updateNumericField("bathrooms", value)}
              />

              <Field
                label="Andar"
                value={form.floor}
                disabled={form.property_type === "house"}
                onChange={(value) => updateNumericField("floor", value)}
              />

              <Field
                label="Vagas"
                value={form.parking_spaces}
                onChange={(value) => updateNumericField("parking_spaces", value)}
              />

              <Field
                label="Nota do bairro"
                value={form.neighborhood_score}
                onChange={(value) => updateNumericField("neighborhood_score", value)}
              />

              <Field
                label="Condomínio"
                value={form.condo_fee}
                onChange={(value) => updateNumericField("condo_fee", value)}
              />

              <Field
                label="Idade do imóvel"
                value={form.age_years}
                onChange={(value) => updateNumericField("age_years", value)}
              />

              <Field
                label="Distância ao centro (km)"
                value={form.distance_to_center_km}
                onChange={(value) => updateNumericField("distance_to_center_km", value)}
              />
            </div>

            <div className="details model-details">
              <p><strong>Valor de mercado estimado:</strong> {formatBRL(estimatedMarketValue)}</p>
              <p>
                <strong>Payload:</strong> property_type, neighborhood, base_price_m2
                e variáveis físicas/financeiras são enviados para <strong>/predict</strong>.
              </p>
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
                  <button
                    className="secondary-button"
                    onClick={() => handleLoadModelInfo()}
                    disabled={modelLoading}
                    type="button"
                  >
                    {modelLoading ? "Consultando modelo..." : "Atualizar modelo/schema"}
                  </button>

                  {modelInfo ? (
                    <div className="details model-details">
                      <p><strong>Status:</strong> {modelInfo.status ?? "loaded"}</p>
                      <p><strong>Model URI:</strong> {modelInfo.model_uri ?? "N/A"}</p>
                      <p><strong>Model name:</strong> {modelInfo.model_name ?? "N/A"}</p>
                      <p><strong>Model version:</strong> {modelInfo.model_version ?? "N/A"}</p>
                      <p><strong>Model alias:</strong> {modelInfo.model_alias ?? "N/A"}</p>
                      <p><strong>Tracking URI:</strong> {modelInfo.tracking_uri ?? "N/A"}</p>
                      <p>
                        <strong>Input columns:</strong>{" "}
                        {(modelInfo.input_columns ?? schemaInfo?.model_input_columns ?? []).join(", ") || "N/A"}
                      </p>
                      <p>
                        <strong>Numeric fields:</strong>{" "}
                        {(schemaInfo?.model_numeric_fields ?? []).join(", ") || "N/A"}
                      </p>
                    </div>
                  ) : (
                    <div className="details model-details">
                      <p>Carregando informações do backend...</p>
                    </div>
                  )}
                </>
              )}
            </section>

            <section className="result-card">
              <span className="status">Backend conectado</span>
              <h2>Resultado</h2>

              <div className="price">
                {prediction === null ? "Aguardando previsão" : formatBRL(prediction)}
              </div>

              <div className="details">
                <p><strong>Tipo:</strong> {form.property_type === "house" ? "Casa" : "Apartamento"}</p>
                <p><strong>Cidade:</strong> {form.city}</p>
                <p><strong>Bairro:</strong> {form.neighborhood}</p>
                <p><strong>Preço base m²:</strong> {formatBRL(form.base_price_m2)}</p>
                <p><strong>Endpoint:</strong> /predict</p>
                <p><strong>Modelo:</strong> {modelInfo?.model_name ?? "apartment-price-regression"}</p>
                <p><strong>Versão:</strong> {modelInfo?.model_version ?? "candidate"}</p>
                <p><strong>URI:</strong> {modelInfo?.model_uri ?? "models:/apartment-price-regression@candidate"}</p>
                <p><strong>Runtime:</strong> FastAPI + MLflow PyFunc</p>
              </div>
            </section>

            {lastPayload && (
              <section className="result-card">
                <span className="status">Debug</span>
                <h2>Último payload enviado</h2>
                <pre className="error">{JSON.stringify(lastPayload, null, 2)}</pre>
              </section>
            )}
          </section>
        </div>
      </section>
    </main>
  )
}

function Field({ label, value, onChange, disabled = false }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        step="any"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  )
}

function SelectField({ label, value, options, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export default App
