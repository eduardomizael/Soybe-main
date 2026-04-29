import { ImageIcon, FolderOpenIcon } from "lucide-react";

interface InputModeSelectorProps {
  value: "single" | "batch";
  onChange: (value: "single" | "batch") => void;
}

export function InputModeSelector({ value, onChange }: InputModeSelectorProps) {
  return (
    <div className="space-y-3">
      <label className="text-sm font-semibold text-gray-700">Modo de Avaliação</label>
      <div className="grid grid-cols-2 gap-4">
        {/* Single Image Card */}
        <div
          onClick={() => onChange("single")}
          className={`cursor-pointer rounded-xl border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all duration-300 ${
            value === "single"
              ? "border-green-500 bg-green-50 text-green-700 shadow-md transform scale-[1.02]"
              : "border-gray-100 bg-white text-gray-500 hover:border-green-200 hover:bg-green-50/50"
          }`}
        >
          <ImageIcon className={`w-6 h-6 ${value === "single" ? "text-green-600" : "text-gray-400"}`} />
          <span className="font-medium text-sm text-center">Imagem Única</span>
        </div>

        {/* Batch Image Card */}
        <div
          onClick={() => onChange("batch")}
          className={`cursor-pointer rounded-xl border-2 p-4 flex flex-col items-center justify-center gap-2 transition-all duration-300 ${
            value === "batch"
              ? "border-green-500 bg-green-50 text-green-700 shadow-md transform scale-[1.02]"
              : "border-gray-100 bg-white text-gray-500 hover:border-green-200 hover:bg-green-50/50"
          }`}
        >
          <FolderOpenIcon className={`w-6 h-6 ${value === "batch" ? "text-green-600" : "text-gray-400"}`} />
          <span className="font-medium text-sm text-center">Lote / Pasta</span>
        </div>
      </div>
    </div>
  );
}
