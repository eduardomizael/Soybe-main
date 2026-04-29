import { useState, useEffect } from "react";
import { ZapIcon, TargetIcon, RocketIcon, LayersIcon, SmartphoneIcon, ClockIcon, StarIcon } from "lucide-react";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  version: string | null;
  onVersionChange: (version: string | null) => void;
}

export default function ModelSelector({ value, onChange, version, onVersionChange }: ModelSelectorProps) {
  const [versions, setVersions] = useState<string[]>([]);
  const [isLoadingVersions, setIsLoadingVersions] = useState(false);

  const models = [
    { 
      id: "EfficientNetB0", 
      name: "EfficientNet-B0",
      description: "Desempenho Veloz",
      icon: ZapIcon,
      color: "text-amber-500",
      bgHover: "hover:bg-amber-50" 
    },
    { 
      id: "EfficientNetB2", 
      name: "EfficientNet-B2",
      description: "Equilíbrio Ideal",
      icon: TargetIcon,
      color: "text-blue-500",
      bgHover: "hover:bg-blue-50" 
    },
    { 
      id: "EfficientNetB7", 
      name: "EfficientNet-B7",
      description: "Máxima Precisão",
      icon: StarIcon,
      color: "text-indigo-500",
      bgHover: "hover:bg-indigo-50" 
    },
    { 
      id: "ResNet50", 
      name: "ResNet-50",
      description: "Arquitetura Clássica",
      icon: LayersIcon,
      color: "text-purple-500",
      bgHover: "hover:bg-purple-50" 
    },
    { 
      id: "MobileNetV3", 
      name: "MobileNet-V3",
      description: "Leve e Eficiente",
      icon: SmartphoneIcon,
      color: "text-teal-500",
      bgHover: "hover:bg-teal-50" 
    },
  ];

  // Fetch available versions when a model is selected
  useEffect(() => {
    if (!value) {
      setVersions([]);
      onVersionChange(null);
      return;
    }

    setIsLoadingVersions(true);
    // VITE_API_URL || 8001 fallback
    const baseUrl = import.meta.env.VITE_API_URL || "http://localhost:8001";
    fetch(`${baseUrl}/training/model_versions/${value}`)
      .then(res => res.json())
      .then(data => {
        setVersions(data);
        if (data.length > 0) {
          onVersionChange(data[0]); // Seleciona automaticamente o mais recente (array ordenado reverse=True)
        } else {
          onVersionChange(null);
        }
      })
      .catch(err => {
        console.error("Erro ao buscar versões:", err);
        setVersions([]);
        onVersionChange(null);
      })
      .finally(() => setIsLoadingVersions(false));
  }, [value, onVersionChange]);

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
          <RocketIcon className="w-4 h-4 text-gray-500" />
          Motor de IA Avançado
        </label>
        <div className="grid grid-cols-2 gap-4">
          {models.map((model) => {
            const Icon = model.icon;
            const isSelected = value === model.id;
            return (
              <div
                key={model.id}
                onClick={() => onChange(model.id)}
                className={`cursor-pointer rounded-xl border-2 p-4 flex flex-col gap-1 transition-all duration-300 ${
                  isSelected
                    ? "border-green-500 bg-gradient-to-br from-green-50 to-emerald-50 text-green-900 shadow-md transform scale-[1.02]"
                    : `border-gray-100 bg-white text-gray-500 hover:border-gray-200 ${model.bgHover}`
                }`}
              >
                <div className="flex justify-between items-start mb-1">
                  <Icon className={`w-6 h-6 ${isSelected ? "text-green-600" : model.color}`} />
                  <div className={`w-3 h-3 rounded-full transition-colors ${isSelected ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" : "bg-gray-200"}`}></div>
                </div>
                <span className="font-bold text-sm leading-tight text-gray-800">{model.name}</span>
                <span className="text-xs font-medium text-gray-500">{model.description}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Seletor de Versão (Aparece somente quando há modelo selecionado) */}
      <div className={`transition-all duration-500 overflow-hidden ${value ? "max-h-32 opacity-100" : "max-h-0 opacity-0"}`}>
         <label className="text-sm font-semibold text-gray-700 flex items-center gap-2 mb-2">
          <ClockIcon className="w-4 h-4 text-gray-500" />
          Versão do Treinamento
        </label>
        {isLoadingVersions ? (
          <div className="text-sm text-gray-500 p-3 bg-gray-50 rounded-xl border border-gray-100 animate-pulse">
            Carregando versões disponíveis...
          </div>
        ) : versions.length > 0 ? (
          <select 
            className="w-full p-3 rounded-xl border-2 border-gray-100 bg-white text-gray-700 text-sm font-medium focus:border-green-500 focus:outline-none focus:ring-4 focus:ring-green-500/10 transition-all cursor-pointer"
            value={version || ""}
            onChange={(e) => onVersionChange(e.target.value)}
          >
            {versions.map((v, idx) => {
              // Extrai ano, mes, dia do filename ex: soybean_model_b2_20230501_123000.pth
              const match = v.match(/_(\d{8})_(\d{6})/);
              let displayStr = v;
              if (match) {
                 const d = match[1]; // 20230501
                 const t = match[2]; // 123000
                 displayStr = `${d.substring(6,8)}/${d.substring(4,6)}/${d.substring(0,4)} às ${t.substring(0,2)}:${t.substring(2,4)}:${t.substring(4,6)}`;
              }
              return (
                 <option key={v} value={v}>
                   {idx === 0 ? "★ " : ""}{displayStr} {idx === 0 ? "(Mais Recente)" : ""}
                 </option>
              )
            })}
          </select>
        ) : (
          <div className="text-sm text-amber-600 bg-amber-50 p-3 rounded-xl border border-amber-100 flex items-center gap-2">
            ⚠️ Nenhuma versão customizada encontrada para {value}. Será usado o peso base do projeto (se existir).
          </div>
        )}
      </div>
    </div>
  );
}