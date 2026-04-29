import { Card, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import { CheckCircle2Icon, AlertCircleIcon, ShieldCheckIcon, ActivityIcon, FingerprintIcon } from "lucide-react";

export interface ClassificationResult {
  filename: string;
  imageUrl: string;
  classification: string;
  confidence: number;
  details: {
    category: string;
    quality: "Excelente" | "Boa" | "Regular" | "Ruim";
    defects?: string[];
  };
}

interface ClassificationResultsProps {
  results: ClassificationResult[];
}

export function ClassificationResults({ results }: ClassificationResultsProps) {
  if (results.length === 0) {
    return null;
  }

  const getQualityColor = (quality: string) => {
    switch (quality) {
      case "Excelente":
        return "bg-emerald-100 text-emerald-800 border-emerald-200";
      case "Boa":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "Regular":
        return "bg-amber-100 text-amber-800 border-amber-200";
      case "Ruim":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getQualityProgressColor = (quality: string) => {
    switch (quality) {
      case "Excelente": return "bg-emerald-500";
      case "Boa": return "bg-blue-500";
      case "Regular": return "bg-amber-500";
      case "Ruim": return "bg-red-500";
      default: return "bg-gray-500";
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between bg-white px-6 py-4 rounded-2xl shadow-sm border border-gray-100">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-green-100 rounded-lg">
            <ShieldCheckIcon className="w-6 h-6 text-green-700" />
          </div>
          <div>
            <h3 className="font-bold text-gray-800 leading-tight">Laudo Concluído</h3>
            <p className="text-sm text-gray-500">Imagens processadas com sucesso</p>
          </div>
        </div>
        <Badge variant="secondary" className="bg-gray-100 text-gray-700 font-bold px-3 py-1 shadow-inner text-sm">
          {results.length} {results.length === 1 ? "Análises" : "Análises"}
        </Badge>
      </div>

      <div className={`grid gap-6 ${results.length === 1 ? "grid-cols-1 max-w-2xl mx-auto" : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"}`}>
        {results.map((result, index) => (
          <Card key={index} className="overflow-hidden bg-white/80 backdrop-blur-md hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 border-gray-100/50">
            <div className="relative h-48 sm:h-56 group overflow-hidden">
              <img
                src={result.imageUrl}
                alt={result.filename}
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent pointer-events-none" />
              <div className="absolute top-3 left-3 flex gap-2">
                <Badge className={`px-2 py-0.5 shadow-sm border ${getQualityColor(result.details.quality)}`}>
                  {result.details.quality}
                </Badge>
              </div>
              <div className="absolute bottom-3 left-3 right-3 flex justify-between items-end">
                <div className="text-white">
                  <span className="block text-xs font-medium text-gray-300 mb-0.5 break-all line-clamp-1">{result.filename}</span>
                  <span className="font-bold text-lg leading-tight flex items-center gap-1.5">
                    <FingerprintIcon className="w-4 h-4" />
                    {result.classification}
                  </span>
                </div>
                {result.details.quality === "Excelente" || result.details.quality === "Boa" ? (
                  <CheckCircle2Icon className="w-6 h-6 text-emerald-400 drop-shadow-md" />
                ) : (
                  <AlertCircleIcon className="w-6 h-6 text-red-500 drop-shadow-md" />
                )}
              </div>
            </div>

            <CardContent className="p-5 space-y-4">
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-sm">
                  <span className="font-medium text-gray-600 flex items-center gap-1.5">
                    <ActivityIcon className="w-4 h-4 text-purple-500" />
                    Confiança do Modelo
                  </span>
                  <span className="font-bold text-gray-800">{result.confidence}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden shadow-inner">
                  <div
                    className={`h-2.5 rounded-full ${getQualityProgressColor(result.details.quality)} transition-all duration-1000 ease-out`}
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {result.details.defects && result.details.defects.length > 0 && result.classification !== "Intact soybeans" && (
                <div className="pt-2 border-t border-gray-100">
                  <p className="text-xs font-bold text-gray-400 uppercase mb-2">Características Específicas</p>
                  <div className="flex flex-wrap gap-1.5">
                    {result.details.defects.map((defect, i) => (
                      <Badge key={i} variant="outline" className="bg-red-50 text-red-700 border-red-200 text-xs px-2 py-0.5">
                        {defect}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
