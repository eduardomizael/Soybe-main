// Array files comporta multiplas imagens
export async function handleClassify(model: string, weight_filename: string | null, files: File[], signal: AbortSignal) {
    if (!model || files.length === 0) {
        throw new Error("Modelo ou arquivos ausentes");
    }

    console.log("Modelo selecionado:", model);
    if (weight_filename) console.log("Versão selecionada:", weight_filename);
    console.log("Arquivos para classificação:", files);

    const formData = new FormData();
    formData.append("model_name", model);
    if (weight_filename) {
        formData.append("weight_filename", weight_filename);
    }
    files.forEach((f) => formData.append("files", f));

    // Chama o endpoint de inferência do backend
    // Implementando VITE_API_URL
    const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8001";
    const response = await fetch("http://localhost:8001/inferencia", {
        method: "POST",
        body: formData,
        signal,
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Erro na requisição: ${response.status} ${text}`);
    }

    return response.json();
}