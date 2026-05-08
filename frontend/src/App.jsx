import { useMemo, useState } from "react"
import "./App.css"

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
  condo_fee: 850.0,
  age_years: 5,
  distance_to_center_km: 4.5,
}

function App() {
  const [form, setForm] = useState(initialForm)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [schemaInfo, setSchemaInfo] = useState(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [showModelInfo, setShowModelInfo] = useState(false)

  const estimatedMarketValue = useMemo(() => {
    return Number(form.base_price_m2 || 0) * Number(form.area_m2 || 0)
  }, [form.base_price_m2, form.area_m2])

  async function handlePredict() {
    setLoading(true)
    setError("")
    setPrediction(null)

    const payload = {
      dataframe_records: [
        {
          property_type: form.property_type,
          city: form.city,
          neighborhood: form.neighborhood,
          base_price_m2: Number(form.base_price_m2),
          area_m2: Number(form.area_m2),
          bedrooms: Number(form.bedrooms),
          bathrooms: Number(form.bathrooms),
          floor: Number(form.floor),
          parking_spaces: Number(form.parking_spaces),
          neighborhood_score: Number(form.neighborhood_score),
          condo_fee: Number(form.condo_fee),
          age_years: Number(form.age_years),
          distance_to_center_km: Number(form.distance_to_center_km),
        },
      ],
    }

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json; charset=utf-8",
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(await response.text())
      }

      const data = await response.json()
      setPrediction(data.predictions[0])
      setModelInfo({
        status: "loaded",
        model_uri: data.model_uri,
        model_name: data.model_name,
        model_version: data.model_version,
        model_alias: data.model_alias,
      })
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
      const [modelResponse, schemaResponse] = await Promise.all([
        fetch("/model"),
        fetch("/schema"),
      ])

      if (!modelResponse.ok) {
        throw new Error(await modelResponse.text())
      }

      if (!schemaResponse.ok) {
        throw new Error(await schemaResponse.text())
      }

      const modelData = await modelResponse.json()
      const schemaData = await schemaResponse.json()

      setModelInfo(modelData)
      setSchemaInfo(schemaData)
    } catch (err) {
      setError(err.message)
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
          condo_fee: Math.min(Number(prev.condo_fee || 0), 350),
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
            Interface para consultar o modelo de regressão servido no backend FastAPI,
            com artefatos versionados no MLflow Registry, schema validado e rollout canário no Kubernetes.
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
                onChange={(v) => updateTextField("neighborhood", v)}
              />

              <Field
                label="Preço base por m²"
                value={form.base_price_m2}
                onChange={(v) => updateNumericField("base_price_m2", v)}
              />

              <Field label="Área (m²)" value={form.area_m2} onChange={(v) => updateNumericField("area_m2", v)} />
              <Field label="Quartos" value={form.bedrooms} onChange={(v) => updateNumericField("bedrooms", v)} />
              <Field label="Banheiros" value={form.bathrooms} onChange={(v) => updateNumericField("bathrooms", v)} />
              <Field label="Andar" value={form.floor} onChange={(v) => updateNumericField("floor", v)} />
              <Field label="Vagas" value={form.parking_spaces} onChange={(v) => updateNumericField("parking_spaces", v)} />
              <Field label="Nota do bairro" value={form.neighborhood_score} onChange={(v) => updateNumericField("neighborhood_score", v)} />
              <Field label="Condomínio" value={form.condo_fee} onChange={(v) => updateNumericField("condo_fee", v)} />
              <Field label="Idade do imóvel" value={form.age_years} onChange={(v) => updateNumericField("age_years", v)} />
              <Field label="Distância ao centro (km)" value={form.distance_to_center_km} onChange={(v) => updateNumericField("distance_to_center_km", v)} />
            </div>

            <div className="details model-details">
              <p>
                <strong>Valor de mercado estimado:</strong>{" "}
                {new Intl.NumberFormat("pt-BR", {
                  style: "currency",
                  currency: "BRL",
                }).format(estimatedMarketValue)}
              </p>
              <p>
                <strong>Observação:</strong> o preço base por m² é preenchido automaticamente pela cidade,
                mas pode ser ajustado manualmente para simular cenários de mercado.
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
                  <button className="secondary-button" onClick={handleLoadModelInfo} disabled={modelLoading}>
                    {modelLoading ? "Consultando modelo..." : "Carregar modelo ativo"}
                  </button>

                  {modelInfo ? (
                    <div className="details model-details">
                      <p><strong>Status:</strong> {modelInfo.status ?? "loaded"}</p>
                      <p><strong>Model URI:</strong> {modelInfo.model_uri}</p>
                      <p><strong>Model name:</strong> {modelInfo.model_name}</p>
                      <p><strong>Model version:</strong> {modelInfo.model_version ?? "N/A"}</p>
                      <p><strong>Model alias:</strong> {modelInfo.model_alias ?? "N/A"}</p>
                      <p><strong>Model source:</strong> {modelInfo.model_source ?? "N/A"}</p>
                      <p><strong>Tracking URI:</strong> {modelInfo.tracking_uri ?? "N/A"}</p>
                      <p><strong>Input columns:</strong> {(modelInfo.input_columns ?? schemaInfo?.model_input_columns ?? []).join(", ")}</p>
                      <p><strong>Numeric fields:</strong> {(schemaInfo?.model_numeric_fields ?? []).join(", ") || "N/A"}</p>
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
                <p><strong>Tipo:</strong> {form.property_type === "house" ? "Casa" : "Apartamento"}</p>
                <p><strong>Cidade:</strong> {form.city}</p>
                <p><strong>Bairro:</strong> {form.neighborhood}</p>
                <p><strong>Preço base m²:</strong> {new Intl.NumberFormat("pt-BR", {
                  style: "currency",
                  currency: "BRL",
                }).format(Number(form.base_price_m2 || 0))}</p>
                <p><strong>Endpoint:</strong> /predict</p>
                <p><strong>Modelo:</strong> {modelInfo?.model_name ?? "apartment-price-regression"}</p>
                <p><strong>Versão:</strong> {modelInfo?.model_version ?? "candidate"}</p>
                <p><strong>URI:</strong> {modelInfo?.model_uri ?? "models:/apartment-price-regression@candidate"}</p>
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

function SelectField({ label, value, options, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
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
