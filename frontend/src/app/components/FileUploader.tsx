import { useRef, useState } from "react";
import { UploadCloudIcon, ImagePlusIcon, FolderPlusIcon } from "lucide-react";

interface FileUploaderProps {
  mode: "single" | "batch";
  onFilesSelected: (files: FileList) => void;
}

export function FileUploader({ mode, onFilesSelected }: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isHovering, setIsHovering] = useState(false);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsHovering(true);
  };

  const handleDragLeave = () => {
    setIsHovering(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsHovering(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple={mode === "batch"}
        onChange={handleChange}
        className="hidden"
        {...(mode === "batch" ? { webkitdirectory: "", directory: "" } as any : {})}
      />
      
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative flex flex-col items-center justify-center w-full min-h-[160px] p-6 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden ${
          isHovering
            ? "border-green-500 bg-green-50 scale-[1.01] shadow-lg"
            : "border-gray-200 bg-gray-50/50 hover:bg-gray-100/70 hover:border-gray-300"
        }`}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-green-50/20 to-emerald-100/20 pointer-events-none"></div>
        <UploadCloudIcon
          className={`w-12 h-12 mb-3 transition-colors duration-300 ${isHovering ? "text-green-500" : "text-gray-400"}`}
        />
        <h3 className="text-sm font-semibold text-gray-700 mb-1 z-10 text-center">
          {mode === "single" ? "Arraste uma imagem ou clique para selecionar" : "Arraste imagens ou selecione uma pasta inteira"}
        </h3>
        <p className="text-xs text-gray-500 z-10 text-center flex items-center justify-center gap-1 mt-2 bg-white/70 px-3 py-1 rounded-full border border-gray-100">
          {mode === "single" ? (
            <><ImagePlusIcon className="w-3 h-3" /> JPG, PNG, WEBP permitidos</>
          ) : (
            <><FolderPlusIcon className="w-3 h-3" /> Processamento em Lote otimizado</>
          )}
        </p>
      </div>
    </div>
  );
}
