import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Home, TrendingUp, CircleDollarSign, Server, Clock3, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";

const initialForm = {
  area_m2: 85,
  bedrooms: 3,
  bathrooms: 2,
  floor: 8,
  parking_spaces: 1,
  neighborhood_score: 8.2,
  condo_fee: 650,
  age_years: 7,
  distance_to_center_km: 5.4,
};

function currencyBRL(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
    maximumFractionDigits: 2,
  }).format(value);
}

function numberValue(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export default function ApartmentPriceFrontend() {
  const [apiUrl, setApiUrl] = useState("http://127.0.0.1:8081/invocations");
  const [form, setForm] = useState(initialForm);
  const [prediction, setPrediction] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const payload = useMemo(() => ({ dataframe_records: [form] }), [form]);

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: numberValue(value) }));
  }

  async function handlePredict(e) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const startedAt = performance.now();

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Falha na predição (${response.status}): ${text || "sem detalhes"}`);
      }

      const data = await response.json();
      const elapsed = Math.round(performance.now() - startedAt);
      const predictedValue = Array.isArray(data.predictions) ? data.predictions[0] : null;

      setPrediction(predictedValue);
      setLatencyMs(elapsed);
      setHistory((prev) => [
        {
          id: crypto.randomUUID(),
          timestamp: new Date().toLocaleString("pt-BR"),
          price: predictedValue,
          latencyMs: elapsed,
          input: { ...form },
        },
        ...prev,
      ].slice(0, 6));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado ao consultar a API.");
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setForm(initialForm);
    setPrediction(null);
    setLatencyMs(null);
    setError("");
  }

  const summary = [
    { label: "Área", value: `${form.area_m2} m²`, icon: Home },
    { label: "Quartos", value: String(form.bedrooms), icon: TrendingUp },
    { label: "Taxa de condomínio", value: currencyBRL(form.condo_fee), icon: CircleDollarSign },
    { label: "API", value: apiUrl.replace("/invocations", ""), icon: Server },
  ];

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <Card className="rounded-2xl border-0 shadow-lg">
            <CardHeader className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-slate-900 p-3 text-white">
                  <Home className="h-6 w-6" />
                </div>
                <div>
                  <CardTitle className="text-2xl md:text-3xl">Previsão de preço de apartamentos</CardTitle>
                  <CardDescription>
                    Frontend simples para consultar o modelo servido pelo MLflow e exibir a estimativa de forma apresentável.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={handlePredict} className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="apiUrl">Endpoint da API</Label>
                    <Input
                      id="apiUrl"
                      value={apiUrl}
                      onChange={(e) => setApiUrl(e.target.value)}
                      placeholder="http://127.0.0.1:8081/invocations"
                    />
                  </div>

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

                {error ? (
                  <Alert className="border-red-200 bg-red-50 text-red-900">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Falha ao consultar a API</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                ) : null}

                <div className="flex flex-wrap gap-3">
                  <Button type="submit" disabled={loading} className="rounded-2xl px-6">
                    {loading ? "Consultando..." : "Calcular previsão"}
                  </Button>
                  <Button type="button" variant="outline" onClick={resetForm} className="rounded-2xl">
                    Limpar
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }} className="space-y-6">
          <Card className="rounded-2xl border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="text-xl">Resumo da consulta</CardTitle>
              <CardDescription>Visualização rápida dos principais parâmetros da predição.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              {summary.map((item) => {
                const Icon = item.icon;
                return (
                  <div key={item.label} className="rounded-2xl border bg-white p-4 shadow-sm">
                    <div className="mb-2 flex items-center gap-2 text-slate-600">
                      <Icon className="h-4 w-4" />
                      <span className="text-sm">{item.label}</span>
                    </div>
                    <div className="text-lg font-semibold text-slate-900">{item.value}</div>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="text-xl">Resultado</CardTitle>
              <CardDescription>Resposta retornada pelo endpoint do modelo.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-2xl bg-slate-900 p-6 text-white shadow-sm">
                <div className="text-sm text-slate-300">Preço previsto</div>
                <div className="mt-2 text-3xl font-bold md:text-4xl">
                  {prediction == null ? "Aguardando consulta" : currencyBRL(prediction)}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-slate-300">
                  <Badge variant="secondary" className="rounded-xl bg-white/10 text-white hover:bg-white/10">
                    MLflow Serving
                  </Badge>
                  {latencyMs != null ? (
                    <span className="inline-flex items-center gap-1">
                      <Clock3 className="h-4 w-4" /> {latencyMs} ms
                    </span>
                  ) : null}
                </div>
              </div>

              <div>
                <div className="mb-2 text-sm font-medium text-slate-700">Payload enviado</div>
                <pre className="overflow-auto rounded-2xl bg-slate-950 p-4 text-xs text-slate-100">
{JSON.stringify(payload, null, 2)}
                </pre>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55 }} className="mx-auto mt-6 max-w-7xl">
        <Card className="rounded-2xl border-0 shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl">Histórico recente</CardTitle>
            <CardDescription>Últimas previsões realizadas na interface atual.</CardDescription>
          </CardHeader>
          <CardContent>
            {history.length === 0 ? (
              <div className="rounded-2xl border border-dashed p-8 text-center text-slate-500">
                Nenhuma previsão realizada ainda.
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {history.map((item) => (
                  <div key={item.id} className="rounded-2xl border bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-sm font-medium text-slate-700">{item.timestamp}</div>
                      <Badge className="rounded-xl">{item.latencyMs} ms</Badge>
                    </div>
                    <Separator className="my-3" />
                    <div className="text-2xl font-bold text-slate-900">{currencyBRL(item.price)}</div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-sm text-slate-600">
                      <div>Área: {item.input.area_m2} m²</div>
                      <div>Quartos: {item.input.bedrooms}</div>
                      <div>Banheiros: {item.input.bathrooms}</div>
                      <div>Andar: {item.input.floor}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

function Field({ label, value, onChange }) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}
