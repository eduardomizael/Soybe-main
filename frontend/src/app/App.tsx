import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import ModelSelector from "./components/ModelSelector";
import { InputModeSelector } from "./components/InputModeSelector";
import { FileUploader } from "./components/FileUploader";
import { ClassificationResults, ClassificationResult } from "./components/ClassificationResults";
import { Navbar } from "./components/Navbar";
import { Dashboard } from "./components/Dashboard";
import { TrainingDashboard } from "./components/TrainingDashboard";
import { Button } from "./components/ui/button";
import { Loader2Icon, SproutIcon, StopCircleIcon, Trash } from "lucide-react";
import { toast } from "sonner";
import { handleClassify } from "./components/ModelPipeline";

type ApiResultItem = {
  index?: number;
  filename?: string;
  classification?: string;
  confidence?: number;
  model_used?: string;
};

// Estado global da tela
function App() {
  const [activeTab, setActiveTab] = useState<"classifier" | "dashboard" | "training">("classifier");
  const [selectedModel, setSelectedModel] = useState(""); // Modelo treinado selecionado
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null); // Versão do modelo selecionado
  const [inputMode, setInputMode] = useState<"single" | "batch">("single"); // Modo de entrada de arquivos
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]); // Arquivos selecionados pelo usuário
  const [results, setResults] = useState<ClassificationResult[]>([]); // Resultados da classificação
  const [isClassifying, setIsClassifying] = useState(false); // Classificação em execução: Sim ou Não
  // Estado para armazenar o controller de cancelamento da requisição
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  
// Manipula arquivos selecionados
  const handleFilesSelected = (files: FileList) => {
    const fileArray = Array.from(files);
    setSelectedFiles(fileArray);
    toast.success(`${fileArray.length} arquivo(s) selecionado(s)`);
  };

  const runClassification = async () => {

    const controller = new AbortController();
    // Salvar oo estado do controller
    setAbortController(controller);

    // Spinner de carregamento
    setIsClassifying(true);

    if (!selectedModel || selectedFiles.length === 0) return;
    setIsClassifying(true);

    try {

      /*
      ModelPipeline (handleClassify) recebe o modelo e arquivos para classificação
      
      Data retorna a estrutura:
      {
        "filename": "image1.jpg",
        "classification": "Intact soybeans",
        "confidence": 0.98,
        "model_used": "EfficientNetB0"
      }

      data.map(item => item.filename, item.classification, item.confidence, item.model_used)

      */
      
      const data = await handleClassify(selectedModel, selectedVersion, selectedFiles, controller.signal);

      const sortedData = data.sort((a: ApiResultItem, b: ApiResultItem) => (a.index ?? 0) - (b.index ?? 0));

      console.log("Data:", sortedData);

      // Resposta do backend estruturada
      const transformedResults = sortedData.map((item: ApiResultItem) => {
        const indexFromApi = item.index ?? -1;

        // Mapeamento principal: usa o índice original enviado pelo backend.
        let correspondingFile = indexFromApi >= 0 ? selectedFiles[indexFromApi] : undefined;

        // Fallback para cenários sem índice: compara apenas o basename do arquivo.
        if (!correspondingFile && item.filename) {
          const responseBaseName = item.filename.split("/").pop() ?? item.filename;
          correspondingFile = selectedFiles.find((f: File) => f.name === responseBaseName);
        }

        if (!correspondingFile) {
          console.warn(`Arquivo ${item.filename ?? "desconhecido"} não encontrado em selectedFiles`);
          return null;
        }

        return {
          filename: item.filename,
          imageUrl: URL.createObjectURL(correspondingFile),
          classification: item.classification,
          confidence: Math.round((item.confidence ?? 0) * 100),
          details: {
            category: item.model_used ?? "N/A",
            quality: getQualityFromConfidence(item.confidence ?? 0),
            defects: item.classification ? [item.classification] : []
          }
        };
      }).filter(Boolean);

      console.log("Transformed Results:", transformedResults);

      setResults(transformedResults);
    } catch (err: any) {
      if (err.name === "AbortError") {
        toast.info("Classificação cancelada pelo usuário");
      } else {
        console.error("Erro:", err);
      }
    } finally {
      setIsClassifying(false);
    }
  };

// Função para chamar pelo botão "Parar"
const stopClassification = () => {
  // Cancelar a requisição HTTP se o controller existir
  abortController?.abort();
  setIsClassifying(false);
};

// Helper para mapear confidence -> quality
function getQualityFromConfidence(confidence: number): "Excelente" | "Boa" | "Regular" | "Ruim" {
  if (confidence >= 0.95) return "Excelente";
  if (confidence >= 0.85) return "Boa";
  if (confidence >= 0.70) return "Regular";
  return "Ruim";
}

// Função para limpar a seleção de arquivos e resultados
const clearSelection = () => {
  setSelectedFiles([]);
  setResults([]);
  toast.success("Seleção limpa");
}

// Função

  // Manipula o clique no botão de classificar (handleClassify movida para ModelPipeline.tsx)

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex flex-col">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} hasResults={results.length > 0} />
      <div className="flex-1 p-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {activeTab === "classifier" ? (
            <>
              {/* Header */}
              <div className="text-center space-y-4 my-8">
                <div className="flex items-center justify-center gap-3">
                  <div className="p-3 bg-white rounded-2xl shadow-sm border border-green-100">
                    <SproutIcon className="w-10 h-10 text-green-600" />
                  </div>
                  <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-800 to-emerald-600 tracking-tight">Soybe System</h1>
                </div>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto font-medium">
                  Visão Computacional e Inteligência Artificial para classificação de qualidade dos grãos de soja.
                </p>
              </div>

        {/* Configuration Panel */}
        <Card className="bg-white/80 backdrop-blur-xl border border-white max-w-4xl mx-auto shadow-xl rounded-3xl overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-green-600 to-emerald-800 text-white p-8">
            <CardTitle className="text-2xl">Painel de Avaliação</CardTitle>
            <CardDescription className="text-green-50 font-medium">
              Configure o motor e faça o envio das imagens para análise
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8 p-8">
            <div className="grid md:grid-cols-2 gap-8 items-start">
              <ModelSelector value={selectedModel} onChange={setSelectedModel} version={selectedVersion} onVersionChange={setSelectedVersion} />
              <InputModeSelector value={inputMode} onChange={setInputMode} />
            </div>

            <div className="h-px w-full bg-gradient-to-r from-transparent via-gray-200 to-transparent"></div>

            <div className="space-y-4">
              <div className="flex justify-between items-center mb-2">
                <h3 className="font-semibold text-gray-700">Arquivos para Análise</h3>
                {selectedFiles.length > 0 && (
                  <Button variant="outline" size="sm" onClick={clearSelection} className="text-red-500 hover:text-red-700 hover:bg-red-50 hover:border-red-200">
                    <Trash className="w-4 h-4 mr-2" />
                    Limpar Seleção ({selectedFiles.length})
                  </Button>
                )}
              </div>
              <FileUploader mode={inputMode} onFilesSelected={handleFilesSelected} />
            </div>

            <div className="flex gap-4 pt-4">
              <Button
                onClick={runClassification}
                disabled={isClassifying || !selectedModel || selectedFiles.length === 0}
                className="flex-1 h-14 text-lg bg-green-600 hover:bg-green-700 shadow-lg shadow-green-600/30 transition-all rounded-xl font-bold"
              >
                {isClassifying ? (
                  <>
                    <Loader2Icon className="w-6 h-6 mr-2 animate-spin" />
                    Processando Imagens...
                  </>
                ) : (
                  "Iniciar Classificação Automática"
                )}
              </Button>

              {isClassifying && (
                <Button
                  onClick={stopClassification}
                  className="h-14 px-8 bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/30 transition-all rounded-xl font-bold text-white flex gap-2"
                >
                  <StopCircleIcon className="w-6 h-6" /> Cancelar
                </Button>
              )}
            </div>

          </CardContent>
        </Card>

        {/* Results Section */}
        {results.length > 0 && (
          <Card>
            <CardContent className="pt-6">
              <ClassificationResults results={results} />
            </CardContent>
          </Card>
        )}
            </>
          ) : activeTab === "dashboard" ? (
            <Dashboard results={results} />
          ) : (
            <TrainingDashboard />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
