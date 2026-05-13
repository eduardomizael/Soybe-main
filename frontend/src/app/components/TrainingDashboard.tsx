import { useState, useEffect, useRef, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import { Button } from "./ui/button";
import { toast } from "sonner";
import {
  BrainCircuitIcon,
  TimerIcon,
  ZapIcon,
  TrendingUpIcon,
  PlayIcon,
  SquareIcon,
  FolderOpenIcon,
  Loader2Icon,
  CheckCircle2Icon,
  XCircleIcon,
  SettingsIcon,
  DatabaseIcon,
  CpuIcon,
  PauseIcon,
  SaveIcon,
} from "lucide-react";

const API_HOST = (import.meta.env.VITE_API_URL || "http://localhost:8001").replace(/\/$/, "");
const API_BASE = `${API_HOST}/training`;
const WS_URL = `${API_HOST.replace(/^http/, "ws")}/training/ws`;

// ──────────────────── Types ────────────────────

interface ModelInfo {
  id: string;
  name: string;
  input_size: number;
  default_batch: number;
  cpu_batch: number;
}

interface DatasetInfo {
  name: string;
  path: string;
  labels: Record<string, number>;
  total_images: number;
  root_images?: number;
}

interface EpochData {
  epoch: number;
  train_loss: number;
  val_loss: number;
  elapsed_seconds?: number;
}

interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

interface TrainingResult {
  total_time: number;
  best_val_loss: number;
  accuracy: number;
  classification_report: PerClassMetric[];
  model_path: string;
  num_classes: number;
  class_names: string[];
  confusion_matrix?: number[][];
  roc_curves?: {
    class: string;
    tpr: number[];
    auc: number;
  }[];
  common_fpr?: number[];
}

interface CancelInfo {
  epochs_completed: number;
  has_checkpoint: boolean;
  checkpoint_path: string | null;
}

type TrainingPhase = "config" | "training" | "completed" | "error" | "cancelled";

// ──────────────────── Component ────────────────────

export function TrainingDashboard() {
  // Config state
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [batchSize, setBatchSize] = useState(16);
  const [numEpochs, setNumEpochs] = useState(20);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [patience, setPatience] = useState(5);
  const [trainSplit, setTrainSplit] = useState(0.8);
  const [valSplit, setValSplit] = useState(0.1);

  // Training state
  const [phase, setPhase] = useState<TrainingPhase>("config");
  const [isPaused, setIsPaused] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [currentBatch, setCurrentBatch] = useState(0);
  const [totalBatches, setTotalBatches] = useState(0);
  const [epochHistory, setEpochHistory] = useState<EpochData[]>([]);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [cancelInfo, setCancelInfo] = useState<CancelInfo | null>(null);

  const [activeTab, setActiveTab] = useState<"new" | "history">("new");
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [expandedHistoryId, setExpandedHistoryId] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // ── Fetch models and datasets on mount ──
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then((r) => r.json())
      .then((data) => setModels(data))
      .catch(() => toast.error("Erro ao carregar modelos"));

    fetch(`${API_BASE}/datasets`)
      .then((r) => r.json())
      .then((data) => setDatasets(data))
      .catch(() => toast.error("Erro ao carregar datasets"));
  }, []);

  useEffect(() => {
    if (activeTab === "history") {
      fetch(`${API_BASE}/history`)
        .then((r) => r.json())
        .then((data) => setHistoryData(data))
        .catch(() => toast.error("Erro ao carregar histórico"));
    }
  }, [activeTab]);

  // Fix 5: Cleanup WebSocket on component unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  // Continuous timer during training
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (phase === "training" && trainingStartTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - trainingStartTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [phase, trainingStartTime]);

  // Update batch_size default when model changes
  useEffect(() => {
    const model = models.find((m) => m.id === selectedModel);
    if (model) {
      setBatchSize(model.default_batch);
    }
  }, [selectedModel, models]);

  // ── WebSocket management ──
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case "status":
          setStatusMessage(msg.message);
          break;

        case "batch_progress":
          setCurrentEpoch(msg.epoch);
          setTotalEpochs(msg.total_epochs);
          setCurrentBatch(msg.batch);
          setTotalBatches(msg.total_batches);
          break;

        case "epoch_complete":
          setCurrentEpoch(msg.epoch);
          setTotalEpochs(msg.total_epochs);
          setCurrentBatch(0);
          setTotalBatches(0);
          setEpochHistory((prev) => [
            ...prev,
            {
              epoch: msg.epoch,
              train_loss: msg.train_loss,
              val_loss: msg.val_loss,
              elapsed_seconds: msg.elapsed_seconds,
            },
          ]);
          break;

        case "training_complete":
          setPhase("completed");
          setResult(msg as TrainingResult);
          toast.success("Treinamento concluído com sucesso!");
          break;

        case "training_cancelled":
          setPhase("cancelled");
          setCancelInfo({
            epochs_completed: msg.epochs_completed ?? 0,
            has_checkpoint: msg.has_checkpoint ?? false,
            checkpoint_path: msg.checkpoint_path ?? null,
          });
          toast.info("Treinamento cancelado.");
          break;

        case "training_error":
          setPhase("error");
          setErrorMessage(msg.message);
          toast.error("Erro no treinamento: " + msg.message);
          break;
      }
    };

    ws.onerror = () => {
      toast.error("Erro na conexão WebSocket");
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, []);

  const disconnectWs = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  // ── Start training ──
  const handleStart = async () => {
    if (!selectedModel || !selectedDataset) {
      toast.error("Selecione um modelo e um dataset.");
      return;
    }

    // Fix 2: Validate splits on client side
    if (trainSplit + valSplit >= 0.95) {
      toast.error(
        `Train (${(trainSplit * 100).toFixed(0)}%) + Val (${(valSplit * 100).toFixed(0)}%) = ${((trainSplit + valSplit) * 100).toFixed(0)}%. A soma deve ser < 95%.`
      );
      return;
    }

    const dataset = datasets.find((d) => d.name === selectedDataset);
    if (!dataset) return;

    // Reset state
    setPhase("training");
    setEpochHistory([]);
    setCurrentEpoch(0);
    setCurrentBatch(0);
    setTotalBatches(0);
    setElapsedTime(0);
    setResult(null);
    setErrorMessage("");
    setCancelInfo(null);
    setStatusMessage("Iniciando treinamento...");
    setTrainingStartTime(Date.now());

    // Connect WebSocket first
    connectWs();

    // Small delay to let WS connect
    await new Promise((r) => setTimeout(r, 500));

    try {
      const res = await fetch(`${API_BASE}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: selectedModel,
          data_path: dataset.path,
          batch_size: batchSize,
          num_epochs: numEpochs,
          learning_rate: learningRate,
          patience: patience,
          train_split: trainSplit,
          val_split: valSplit,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Erro ao iniciar treinamento");
      }

      toast.success("Treinamento iniciado!");
    } catch (err: any) {
      setPhase("error");
      setErrorMessage(err.message);
      setTrainingStartTime(null);
      toast.error(err.message);
      disconnectWs();
    }
  };

  // ── Pause training ──
  const handlePauseToggle = async () => {
    try {
      const endpoint = isPaused ? "resume" : "pause";
      await fetch(`${API_BASE}/${endpoint}`, { method: "POST" });
      setIsPaused(!isPaused);
      toast.info(isPaused ? "Treinamento retomado!" : "Treinamento pausado.");
    } catch {
      toast.error("Erro ao alternar pausa do treinamento");
    }
  };

  // ── Stop early ──
  const handleStopEarly = async () => {
    try {
      await fetch(`${API_BASE}/stop_early`, { method: "POST" });
      setIsPaused(false);
      toast.info("Finalizando antecipadamente. Salvando a época atual...");
    } catch {
      toast.error("Erro ao finalizar treinamento");
    }
  };

  // ── Cancel training ──
  const handleCancel = async () => {
    try {
      await fetch(`${API_BASE}/cancel`, { method: "POST" });
      toast.info("Cancelamento solicitado...");
    } catch {
      toast.error("Erro ao cancelar treinamento");
    }
  };

  // ── Reset to config ──
  const handleReset = () => {
    setPhase("config");
    setIsPaused(false);
    setEpochHistory([]);
    setCurrentEpoch(0);
    setCurrentBatch(0);
    setResult(null);
    setErrorMessage("");
    setCancelInfo(null);
    setStatusMessage("");
    setTrainingStartTime(null);
    disconnectWs();
  };

  // ── Helpers ──
  const epochProgress = totalEpochs > 0 ? (currentEpoch / totalEpochs) * 100 : 0;
  const batchProgress = totalBatches > 0 ? (currentBatch / totalBatches) * 100 : 0;
  const testSplit = Math.max(0, 1 - trainSplit - valSplit);

  const formatTime = (s: number) => {
    if (s < 60) return `${s.toFixed(0)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  };

  // ──────────────────── Render ────────────────────

  return (
    <div className="space-y-6">
      {/* ═══════════ TABS ═══════════ */}
      <div className="flex items-center gap-2 p-1 bg-gray-100/80 backdrop-blur-md rounded-2xl w-fit border border-gray-200">
        <button
          onClick={() => setActiveTab("new")}
          className={`px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ${
            activeTab === "new"
              ? "bg-white shadow-md text-emerald-700"
              : "text-gray-500 hover:text-gray-800 hover:bg-white/50"
          }`}
        >
          Novo Treinamento
        </button>
        <button
          onClick={() => setActiveTab("history")}
          className={`px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ${
            activeTab === "history"
              ? "bg-white shadow-md text-blue-700"
              : "text-gray-500 hover:text-gray-800 hover:bg-white/50"
          }`}
        >
          Histórico
        </button>
      </div>

      {activeTab === "history" && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2">
          <div className="flex items-center gap-3 mb-2">
            <h2 className="text-2xl font-bold text-gray-800">Modelos Treinados</h2>
            <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-xs font-bold">
              {historyData.length} registros
            </span>
          </div>

          {historyData.length === 0 ? (
            <div className="text-gray-500 italic p-12 text-center border-2 border-dashed border-gray-300 rounded-3xl bg-white/50">
              Nenhum histórico de treinamento encontrado. Realize um treinamento para ver os resultados aqui!
            </div>
          ) : (
            <div className="grid gap-5 xl:grid-cols-2">
              {historyData.map((item, idx) => {
                const isExpanded = expandedHistoryId === idx;
                const rocData = isExpanded && item.result.common_fpr ? item.result.common_fpr.map((fpr: number, fIdx: number) => {
                  const point: any = { fpr: Number(fpr.toFixed(3)) };
                  item.result.roc_curves?.forEach((rc: any) => {
                    point[rc.class] = rc.tpr[fIdx];
                  });
                  return point;
                }) : [];

                return (
                <Card key={idx} className="bg-white/90 backdrop-blur-xl border-gray-200 shadow-lg hover:shadow-xl transition-shadow rounded-3xl overflow-hidden">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50/30 pb-4 border-b border-gray-100">
                    <CardTitle className="text-lg flex justify-between items-center">
                      <span className="text-blue-800 font-extrabold flex items-center gap-2">
                        <CpuIcon className="w-5 h-5" />
                        {item.model_name}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-gray-500 bg-white px-3 py-1 rounded-full shadow-sm">
                          {new Date(item.timestamp).toLocaleString("pt-BR", {
                            day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
                          })}
                        </span>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="h-7 text-xs px-2"
                          onClick={() => setExpandedHistoryId(isExpanded ? null : idx)}
                        >
                          {isExpanded ? "Ocultar" : "Detalhes"}
                        </Button>
                      </div>
                    </CardTitle>
                    <CardDescription className="flex items-center gap-2 mt-1 !text-gray-500 font-medium">
                      <DatabaseIcon className="w-4 h-4" />
                      Dataset: {item.dataset_name}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-5 space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-emerald-50 rounded-xl p-3 border border-emerald-100">
                        <div className="text-[10px] uppercase font-bold text-emerald-600 mb-1 tracking-wider">Acurácia (Teste)</div>
                        <div className="text-xl font-black text-emerald-700">
                          {item.result.accuracy}%
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                        <div className="text-[10px] uppercase font-bold text-gray-500 mb-1 tracking-wider">Val Loss</div>
                        <div className="text-lg font-bold text-gray-800">
                          {item.result.best_val_loss}
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                        <div className="text-[10px] uppercase font-bold text-gray-500 mb-1 tracking-wider">Épocas</div>
                        <div className="text-lg font-bold text-gray-800">
                          {item.config.num_epochs}
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                        <div className="text-[10px] uppercase font-bold text-gray-500 mb-1 tracking-wider">Tempo</div>
                        <div className="text-lg font-bold text-gray-800">
                          {formatTime(item.result.total_time)}
                        </div>
                      </div>
                    </div>
                    
                    <div className="pt-2">
                      <div className="text-[10px] uppercase font-bold text-gray-400 mb-2 tracking-wider">
                        {item.result.num_classes} Classes Detectadas
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {item.result.class_names.map((c: string) => (
                          <span key={c} className="text-xs px-2.5 py-1 bg-white/80 text-gray-700 rounded-lg font-semibold border border-gray-200 shadow-sm">
                            {c}
                          </span>
                        ))}
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="mt-4 pt-4 border-t border-gray-100 space-y-6 animate-in fade-in slide-in-from-top-2">
                        {/* Parameters */}
                        <div>
                          <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Parâmetros de Treinamento</h4>
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                            <div className="bg-gray-50 p-2 rounded-lg text-xs border border-gray-100"><span className="text-gray-500 block mb-0.5" style={{fontSize: '10px'}}>Batch Size</span><span className="font-semibold text-gray-700">{item.config.batch_size}</span></div>
                            <div className="bg-gray-50 p-2 rounded-lg text-xs border border-gray-100"><span className="text-gray-500 block mb-0.5" style={{fontSize: '10px'}}>Épocas (Config)</span><span className="font-semibold text-gray-700">{item.config.num_epochs}</span></div>
                            <div className="bg-gray-50 p-2 rounded-lg text-xs border border-gray-100"><span className="text-gray-500 block mb-0.5" style={{fontSize: '10px'}}>Learning Rate</span><span className="font-semibold text-gray-700">{item.config.learning_rate}</span></div>
                            <div className="bg-gray-50 p-2 rounded-lg text-xs border border-gray-100"><span className="text-gray-500 block mb-0.5" style={{fontSize: '10px'}}>Splits (T/V)</span><span className="font-semibold text-gray-700">{item.config.train_split} / {item.config.val_split}</span></div>
                          </div>
                        </div>

                        {/* Classification Report */}
                        {item.result.classification_report && (
                          <div>
                            <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Relatório de Classificação</h4>
                            <div className="overflow-x-auto rounded-xl border border-gray-100">
                              <table className="w-full text-xs text-center border-collapse">
                                <thead className="bg-gray-50 text-gray-600">
                                  <tr>
                                    <th className="p-2 border-b text-left">Classe</th>
                                    <th className="p-2 border-b">Precision</th>
                                    <th className="p-2 border-b">Recall</th>
                                    <th className="p-2 border-b">F1-Score</th>
                                    <th className="p-2 border-b">Suporte</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {item.result.classification_report.map((row: any, i: number) => (
                                    <tr key={i} className="border-b last:border-0 hover:bg-gray-50/50">
                                      <td className="p-2 font-medium text-left">{row.class}</td>
                                      <td className="p-2 text-emerald-600 font-semibold">{row.precision}%</td>
                                      <td className="p-2 text-blue-600 font-semibold">{row.recall}%</td>
                                      <td className="p-2 text-purple-600 font-semibold">{row.f1}%</td>
                                      <td className="p-2 text-gray-500">{row.support}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Confusion Matrix */}
                        {item.result.confusion_matrix && (
                          <div>
                            <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Matriz de Confusão</h4>
                            <div className="overflow-x-auto rounded-xl border border-gray-100 p-1">
                              <table className="min-w-full text-center text-xs border-collapse">
                                <thead>
                                  <tr>
                                    <th className="p-1.5 text-gray-500 font-medium border-b border-r bg-gray-50/50" style={{fontSize: '10px'}}>Real \ Pred</th>
                                    {item.result.class_names.map((c: string) => (
                                      <th key={c} className="p-1.5 font-medium text-gray-600 border-b border-gray-100 truncate max-w-[80px]" title={c}>{c}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {item.result.confusion_matrix.map((row: number[], rIdx: number) => {
                                    const sumRow = row.reduce((a, b) => a + b, 0);
                                    return (
                                      <tr key={rIdx}>
                                        <th className="p-1.5 font-medium text-gray-600 text-left border-r border-gray-100 truncate max-w-[80px] bg-gray-50/50" title={item.result.class_names[rIdx]}>
                                          {item.result.class_names[rIdx]}
                                        </th>
                                        {row.map((val: number, cIdx: number) => {
                                          const isCorrect = rIdx === cIdx;
                                          const ratio = sumRow > 0 ? val / sumRow : 0;
                                          const alpha = Math.max(0.05, ratio * 0.85);
                                          const bgColor = isCorrect 
                                            ? `rgba(16, 185, 129, ${alpha})` 
                                            : (val > 0 ? `rgba(239, 68, 68, ${alpha})` : 'transparent');
                                          
                                          return (
                                            <td key={cIdx} className="p-1.5 border border-gray-50" style={{ backgroundColor: bgColor }}>
                                              <span className={ratio > 0.5 ? "text-white font-bold drop-shadow-sm" : "text-gray-700 font-medium"}>
                                                {val}
                                              </span>
                                            </td>
                                          );
                                        })}
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* ROC Curve */}
                        {rocData.length > 0 && item.result.roc_curves && (
                          <div>
                            <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Curva ROC</h4>
                            <div className="h-64 mt-2 bg-gray-50/30 rounded-xl border border-gray-100 p-2">
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={rocData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                                  <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.5} />
                                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickCount={6} tickFormatter={(v) => v.toFixed(1)} stroke="#9ca3af" fontSize={10} />
                                  <YAxis type="number" domain={[0, 1]} tickCount={6} tickFormatter={(v) => v.toFixed(1)} stroke="#9ca3af" fontSize={10} />
                                  <Tooltip 
                                    formatter={(val: number) => val.toFixed(3)} 
                                    labelFormatter={(lbl) => `Taxa Falso Positivo: ${lbl}`}
                                    wrapperStyle={{ fontSize: '11px', borderRadius: '8px' }}
                                  />
                                  <Legend wrapperStyle={{ fontSize: '10px' }} />
                                  {item.result.roc_curves.map((rc: any, i: number) => {
                                    const colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"];
                                    return (
                                      <Line 
                                        key={rc.class} 
                                        type="monotone" 
                                        dataKey={rc.class} 
                                        name={`${rc.class} (AUC: ${rc.auc.toFixed(2)})`} 
                                        stroke={colors[i % colors.length]} 
                                        dot={false} 
                                        strokeWidth={2}
                                      />
                                    );
                                  })}
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )})}
            </div>
          )}
        </div>
      )}

      {activeTab === "new" && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2">
          {/* ═══════════ PHASE: CONFIG ═══════════ */}
          {phase === "config" && (
            <>
          {/* Model Selection */}
          <Card className="bg-white/80 backdrop-blur-xl border border-white shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-purple-600 to-indigo-700 text-white p-6">
              <div className="flex items-center gap-3">
                <CpuIcon className="w-6 h-6" />
                <CardTitle className="text-xl">Seleção de Modelo</CardTitle>
              </div>
              <CardDescription className="text-purple-100">
                Escolha o modelo de rede neural para treinar
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model.id)}
                    className={`relative p-5 rounded-2xl border-2 transition-all duration-300 text-left group ${
                      selectedModel === model.id
                        ? "border-purple-500 bg-purple-50 shadow-lg shadow-purple-200 scale-[1.02]"
                        : "border-gray-200 bg-white hover:border-purple-300 hover:shadow-md"
                    }`}
                  >
                    {selectedModel === model.id && (
                      <div className="absolute top-3 right-3">
                        <CheckCircle2Icon className="w-5 h-5 text-purple-600" />
                      </div>
                    )}
                    <CpuIcon
                      className={`w-8 h-8 mb-3 ${
                        selectedModel === model.id
                          ? "text-purple-600"
                          : "text-gray-400 group-hover:text-purple-400"
                      } transition-colors`}
                    />
                    <div className="font-bold text-gray-800 text-sm">
                      {model.name}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Input: {model.input_size}×{model.input_size}
                    </div>
                    <div className="text-xs text-gray-400 mt-0.5">
                      Batch padrão: {model.default_batch}
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Dataset Selection */}
          <Card className="bg-white/80 backdrop-blur-xl border border-white shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-emerald-600 to-teal-700 text-white p-6">
              <div className="flex items-center gap-3">
                <DatabaseIcon className="w-6 h-6" />
                <CardTitle className="text-xl">Dataset</CardTitle>
              </div>
              <CardDescription className="text-emerald-100">
                Selecione o conjunto de dados de treinamento (pastas = labels)
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              {datasets.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <FolderOpenIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Nenhum dataset encontrado em <code>data/</code></p>
                </div>
              ) : (
                <div className="space-y-3">
                  {datasets.map((ds) => {
                    const isValid = Object.keys(ds.labels).length >= 2;
                    return (
                    <button
                      key={ds.name}
                      onClick={() => isValid && setSelectedDataset(ds.name)}
                      disabled={!isValid}
                      className={`w-full p-5 rounded-2xl border-2 transition-all duration-300 text-left ${
                        selectedDataset === ds.name
                          ? "border-emerald-500 bg-emerald-50 shadow-lg shadow-emerald-200"
                          : !isValid
                          ? "border-red-200 bg-red-50/50 cursor-not-allowed"
                          : "border-gray-200 bg-white hover:border-emerald-300 hover:shadow-md"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <FolderOpenIcon
                            className={`w-5 h-5 ${
                              selectedDataset === ds.name
                                ? "text-emerald-600"
                                : "text-gray-400"
                            }`}
                          />
                          <span className="font-bold text-gray-800">
                            {ds.name}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full text-xs font-semibold">
                            {ds.total_images} imagens
                          </span>
                          <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-xs font-semibold">
                            {Object.keys(ds.labels).length} classes
                          </span>
                          {selectedDataset === ds.name && (
                            <CheckCircle2Icon className="w-5 h-5 text-emerald-600" />
                          )}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(ds.labels).map(([label, count]) => (
                          <span
                            key={label}
                            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs ${
                              !isValid ? "bg-red-100/50 text-red-700" : "bg-gray-100 text-gray-700"
                            }`}
                          >
                            <span className="font-medium">{label}</span>
                            <span className={!isValid ? "text-red-500" : "text-gray-400"}>({count})</span>
                          </span>
                        ))}
                      </div>

                      {!isValid && (
                        <div className="mt-3 text-xs text-red-600 bg-red-100/80 p-3 rounded-xl border border-red-200">
                          <strong>⚠️ Dataset Inválido:</strong> Um dataset requer pelo menos 2 pastas de classes (labels).
                          {ds.root_images ? ` Foram encontradas ${ds.root_images} imagens na raiz desta pasta. Você deve agrupá-as dentro de subpastas correspondentes às suas classes (ex: data/${ds.name}/classe_1/).` : ""}
                        </div>
                      )}
                    </button>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Hyperparameters */}
          <Card className="bg-white/80 backdrop-blur-xl border border-white shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-amber-500 to-orange-600 text-white p-6">
              <div className="flex items-center gap-3">
                <SettingsIcon className="w-6 h-6" />
                <CardTitle className="text-xl">Hiperparâmetros</CardTitle>
              </div>
              <CardDescription className="text-amber-100">
                Configure os parâmetros de treinamento
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Batch Size */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-gray-700">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={128}
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-all outline-none text-gray-800 font-mono"
                  />
                  <p className="text-xs text-gray-400">
                    Imagens por iteração (1–128)
                  </p>
                </div>

                {/* Epochs */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-gray-700">
                    Épocas
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={500}
                    value={numEpochs}
                    onChange={(e) => setNumEpochs(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-all outline-none text-gray-800 font-mono"
                  />
                  <p className="text-xs text-gray-400">
                    Passadas completas pelo dataset (1–500)
                  </p>
                </div>

                {/* Learning Rate */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-gray-700">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    min={0.000001}
                    max={0.1}
                    step={0.0001}
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-all outline-none text-gray-800 font-mono"
                  />
                  <p className="text-xs text-gray-400">
                    Taxa de aprendizado (1e-6 a 0.01)
                  </p>
                </div>

                {/* Patience */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-gray-700">
                    Patience (Early Stopping)
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={patience}
                    onChange={(e) => setPatience(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-all outline-none text-gray-800 font-mono"
                  />
                  <p className="text-xs text-gray-400">
                    Épocas sem melhoria antes de parar
                  </p>
                </div>

                {/* Train / Val Split */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-gray-700">
                    Train Split
                  </label>
                  <input
                    type="number"
                    min={0.5}
                    max={0.95}
                    step={0.05}
                    value={trainSplit}
                    onChange={(e) => setTrainSplit(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-all outline-none text-gray-800 font-mono"
                  />
                  <p className="text-xs text-gray-400">
                    Fração para treinamento ({(trainSplit * 100).toFixed(0)}%
                    treino / {(valSplit * 100).toFixed(0)}% val /{" "}
                    {(testSplit * 100).toFixed(0)}% teste)
                  </p>
                </div>

                {/* Val Split */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-gray-700">
                    Validation Split
                  </label>
                  <input
                    type="number"
                    min={0.05}
                    max={0.4}
                    step={0.05}
                    value={valSplit}
                    onChange={(e) => setValSplit(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-all outline-none text-gray-800 font-mono"
                  />
                  <p className="text-xs text-gray-400">
                    Fração para validação
                  </p>
                </div>
              </div>

              {/* Start button */}
              <div className="mt-8 flex justify-center">
                <Button
                  onClick={handleStart}
                  disabled={!selectedModel || !selectedDataset}
                  className="h-14 px-12 text-lg bg-gradient-to-r from-green-600 to-emerald-700 hover:from-green-700 hover:to-emerald-800 shadow-xl shadow-green-600/30 transition-all rounded-2xl font-bold gap-3 disabled:opacity-50"
                >
                  <PlayIcon className="w-6 h-6" />
                  Iniciar Treinamento
                </Button>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* ═══════════ PHASE: TRAINING ═══════════ */}
      {phase === "training" && (
        <>
          {/* Status Header */}
          <Card className="bg-white/80 backdrop-blur-xl border border-white shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-blue-600 to-cyan-700 text-white p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Loader2Icon className={`w-6 h-6 ${isPaused ? '' : 'animate-spin'}`} />
                  <CardTitle className="text-xl">
                    {isPaused ? `Pausado: ${selectedModel}` : `Treinando ${selectedModel}`}
                  </CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handlePauseToggle}
                    variant="outline"
                    size="sm"
                    className="bg-white/10 border-white/30 text-white hover:bg-white/20 gap-2 h-9"
                  >
                    {isPaused ? <PlayIcon className="w-4 h-4" /> : <PauseIcon className="w-4 h-4" />}
                    <span className="hidden sm:inline">{isPaused ? "Retomar" : "Pausar"}</span>
                  </Button>

                  <Button
                    onClick={handleStopEarly}
                    variant="outline"
                    size="sm"
                    className="bg-emerald-500/80 border-emerald-400/50 text-white hover:bg-emerald-500 gap-2 h-9"
                  >
                    <SaveIcon className="w-4 h-4" />
                    <span className="hidden sm:inline">Finalizar & Salvar</span>
                  </Button>

                  <Button
                    onClick={handleCancel}
                    variant="outline"
                    size="sm"
                    className="bg-red-500/80 border-red-400/50 text-white hover:bg-red-500 gap-2 h-9"
                  >
                    <SquareIcon className="w-4 h-4" />
                    <span className="hidden sm:inline">Descartar</span>
                  </Button>
                </div>
              </div>
              {statusMessage && (
                <p className="text-blue-100 text-sm mt-2">{statusMessage}</p>
              )}
            </CardHeader>
            <CardContent className="p-6 space-y-6">
              {/* Epoch progress */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm font-medium text-gray-700">
                  <span>Progresso Geral — Épocas</span>
                  <span>
                    {currentEpoch} / {totalEpochs || numEpochs}
                  </span>
                </div>
                <div className="h-4 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${epochProgress}%` }}
                  />
                </div>
              </div>

              {/* Batch progress */}
              {totalBatches > 0 && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-medium text-gray-600">
                    <span>
                      Epoch {currentEpoch} — Batches
                    </span>
                    <span>
                      {currentBatch} / {totalBatches}
                    </span>
                  </div>
                  <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-400 to-pink-400 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${batchProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Live metrics cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-2xl p-4 text-center">
                  <BrainCircuitIcon className="w-6 h-6 text-purple-500 mx-auto mb-1" />
                  <div className="text-xl font-bold text-gray-800">
                    {currentEpoch}/{totalEpochs || numEpochs}
                  </div>
                  <div className="text-xs text-gray-500 uppercase font-semibold">
                    Época
                  </div>
                </div>
                <div className="bg-gray-50 rounded-2xl p-4 text-center">
                  <TrendingUpIcon className="w-6 h-6 text-red-500 mx-auto mb-1" />
                  <div className="text-xl font-bold text-gray-800">
                    {epochHistory.length > 0
                      ? epochHistory[epochHistory.length - 1].train_loss.toFixed(4)
                      : "—"}
                  </div>
                  <div className="text-xs text-gray-500 uppercase font-semibold">
                    Train Loss
                  </div>
                </div>
                <div className="bg-gray-50 rounded-2xl p-4 text-center">
                  <ZapIcon className="w-6 h-6 text-amber-500 mx-auto mb-1" />
                  <div className="text-xl font-bold text-gray-800">
                    {epochHistory.length > 0
                      ? epochHistory[epochHistory.length - 1].val_loss.toFixed(4)
                      : "—"}
                  </div>
                  <div className="text-xs text-gray-500 uppercase font-semibold">
                    Val Loss
                  </div>
                </div>
                <div className="bg-gray-50 rounded-2xl p-4 text-center">
                  <TimerIcon className="w-6 h-6 text-blue-500 mx-auto mb-1" />
                  <div className="text-xl font-bold text-gray-800">
                    {formatTime(elapsedTime)}
                  </div>
                  <div className="text-xs text-gray-500 uppercase font-semibold">
                    Tempo
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Live Loss Chart */}
          {epochHistory.length > 0 && (
            <Card className="bg-white/80 backdrop-blur-xl border border-white shadow-xl rounded-3xl overflow-hidden">
              <CardHeader>
                <CardTitle>Evolução do Loss (Tempo Real)</CardTitle>
                <CardDescription>
                  Train loss vs Validation loss por época
                </CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={epochHistory}
                    margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                  >
                    <defs>
                      <linearGradient
                        id="colorTrain"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#ef4444"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#ef4444"
                          stopOpacity={0}
                        />
                      </linearGradient>
                      <linearGradient
                        id="colorVal"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#3b82f6"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#3b82f6"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip
                      formatter={(value: number) => value.toFixed(6)}
                    />
                    <Legend
                      layout="horizontal"
                      verticalAlign="bottom"
                      align="center"
                    />
                    <Area
                      type="monotone"
                      dataKey="train_loss"
                      name="Loss Treino"
                      stroke="#ef4444"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#colorTrain)"
                    />
                    <Area
                      type="monotone"
                      dataKey="val_loss"
                      name="Loss Validação"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#colorVal)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* ═══════════ PHASE: COMPLETED ═══════════ */}
      {phase === "completed" && result && (
        <>
          {/* Success header */}
          <Card className="bg-white/80 backdrop-blur-xl border border-white shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-green-600 to-emerald-700 text-white p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CheckCircle2Icon className="w-6 h-6" />
                  <CardTitle className="text-xl">
                    Treinamento Concluído!
                  </CardTitle>
                </div>
                <Button
                  onClick={handleReset}
                  variant="outline"
                  className="bg-white/10 border-white/30 text-white hover:bg-white/20"
                >
                  Novo Treinamento
                </Button>
              </div>
              <CardDescription className="text-green-100 mt-1">
                Modelo {selectedModel} treinado com sucesso em{" "}
                {formatTime(result.total_time)}
              </CardDescription>
            </CardHeader>
          </Card>

          {/* Final metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-white/60 rounded-2xl">
              <CardContent className="pt-6 flex flex-col items-center justify-center text-center space-y-2">
                <ZapIcon className="w-8 h-8 text-yellow-500" />
                <div className="text-2xl font-bold">{result.accuracy}%</div>
                <div className="text-xs text-gray-500 uppercase font-semibold">
                  Acurácia
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/60 rounded-2xl">
              <CardContent className="pt-6 flex flex-col items-center justify-center text-center space-y-2">
                <TrendingUpIcon className="w-8 h-8 text-red-500" />
                <div className="text-2xl font-bold">
                  {result.best_val_loss.toFixed(4)}
                </div>
                <div className="text-xs text-gray-500 uppercase font-semibold">
                  Melhor Val Loss
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/60 rounded-2xl">
              <CardContent className="pt-6 flex flex-col items-center justify-center text-center space-y-2">
                <BrainCircuitIcon className="w-8 h-8 text-purple-500" />
                <div className="text-2xl font-bold">{epochHistory.length}</div>
                <div className="text-xs text-gray-500 uppercase font-semibold">
                  Épocas Treinadas
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/60 rounded-2xl">
              <CardContent className="pt-6 flex flex-col items-center justify-center text-center space-y-2">
                <TimerIcon className="w-8 h-8 text-blue-500" />
                <div className="text-2xl font-bold">
                  {formatTime(result.total_time)}
                </div>
                <div className="text-xs text-gray-500 uppercase font-semibold">
                  Tempo Total
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Loss History Chart */}
          {epochHistory.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Histórico de Erro (Loss)</CardTitle>
                <CardDescription>
                  Evolução do loss durante o treinamento
                </CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={epochHistory}
                    margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip
                      formatter={(value: number) => value.toFixed(6)}
                    />
                    <Legend
                      layout="horizontal"
                      verticalAlign="bottom"
                      align="center"
                    />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      name="Loss Treino"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="Loss Validação"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Classification Report */}
          <Card>
            <CardHeader>
              <CardTitle>
                Relatório de Classificação (Conjunto de Teste)
              </CardTitle>
              <CardDescription>
                Métricas por classe após avaliação no test set
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-center border-collapse">
                  <thead className="bg-gray-100 text-gray-700 uppercase">
                    <tr>
                      <th className="px-6 py-3 border-b text-left">
                        Classe
                      </th>
                      <th className="px-6 py-3 border-b">Precision (%)</th>
                      <th className="px-6 py-3 border-b">Recall (%)</th>
                      <th className="px-6 py-3 border-b">F1-Score (%)</th>
                      <th className="px-6 py-3 border-b">Suporte</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.classification_report.map((row, index) => (
                      <tr
                        key={index}
                        className="border-b hover:bg-gray-50 transition-colors"
                      >
                        <td className="px-6 py-4 font-medium text-left">
                          {row.class}
                        </td>
                        <td className="px-6 py-4 text-emerald-600 font-semibold">
                          {row.precision}%
                        </td>
                        <td className="px-6 py-4 text-blue-600 font-semibold">
                          {row.recall}%
                        </td>
                        <td className="px-6 py-4 text-purple-600 font-semibold">
                          {row.f1}%
                        </td>
                        <td className="px-6 py-4 text-gray-500">
                          {row.support}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Model saved path */}
              <div className="mt-6 p-4 bg-green-50 rounded-xl border border-green-200 flex items-center gap-3">
                <CheckCircle2Icon className="w-5 h-5 text-green-600 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-green-800">
                    Pesos salvos em:
                  </p>
                  <p className="text-xs text-green-600 font-mono">
                    {result.model_path}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* ═══════════ PHASE: ERROR ═══════════ */}
      {phase === "error" && (
        <Card className="bg-white/80 backdrop-blur-xl border border-red-200 shadow-xl rounded-3xl overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-red-500 to-rose-600 text-white p-6">
            <div className="flex items-center gap-3">
              <XCircleIcon className="w-6 h-6" />
              <CardTitle className="text-xl">Erro no Treinamento</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="p-6 space-y-4">
            <div className="p-4 bg-red-50 rounded-xl border border-red-200">
              <p className="text-red-700 font-mono text-sm">{errorMessage}</p>
            </div>
            <Button onClick={handleReset} className="gap-2">
              Voltar para Configuração
            </Button>
          </CardContent>
        </Card>
      )}

      {/* ═══════════ PHASE: CANCELLED ═══════════ */}
      {phase === "cancelled" && (
        <Card className="bg-white/80 backdrop-blur-xl border border-amber-200 shadow-xl rounded-3xl overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-amber-500 to-orange-600 text-white p-6">
            <div className="flex items-center gap-3">
              <SquareIcon className="w-6 h-6" />
              <CardTitle className="text-xl">
                Treinamento Cancelado
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent className="p-6 space-y-4">
            {cancelInfo?.has_checkpoint ? (
              <div className="p-4 bg-green-50 rounded-xl border border-green-200">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2Icon className="w-5 h-5 text-green-600" />
                  <p className="text-sm text-green-800 font-semibold">
                    Checkpoint parcial salvo!
                  </p>
                </div>
                <p className="text-sm text-green-700">
                  {cancelInfo.epochs_completed} época(s) completadas.
                  Melhor modelo salvo em:
                </p>
                <p className="text-xs text-green-600 font-mono mt-1">
                  {cancelInfo.checkpoint_path}
                </p>
              </div>
            ) : (
              <p className="text-gray-600">
                O treinamento foi interrompido pelo usuário.
                {cancelInfo && cancelInfo.epochs_completed > 0
                  ? ` ${cancelInfo.epochs_completed} época(s) foram executadas, mas nenhum checkpoint melhorou o loss inicial.`
                  : " Nenhum modelo foi salvo."}
              </p>
            )}

            {/* Show partial loss chart if any epochs completed */}
            {epochHistory.length > 0 && (
              <div className="p-4 bg-amber-50 rounded-xl border border-amber-200">
                <p className="text-sm text-amber-800 font-medium">
                  {epochHistory.length} época(s) completadas com dados de loss registrados.
                </p>
              </div>
            )}

            <Button onClick={handleReset} className="gap-2">
              Nova Configuração
            </Button>
          </CardContent>
        </Card>
      )}
        </div>
      )}
    </div>
  );
}
