import { useMemo } from "react";
import { ClassificationResult } from "./ClassificationResults";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, BarChart, XAxis, YAxis, Bar, CartesianGrid } from "recharts";
import { ActivityIcon, LayersIcon, PieChartIcon, TargetIcon } from "lucide-react";

interface DashboardProps {
  results: ClassificationResult[];
}

const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

export function Dashboard({ results }: DashboardProps) {
  const stats = useMemo(() => {
    if (results.length === 0) return null;

    // Calcular estatísticas
    const total = results.length;
    const avgConfidence = results.reduce((sum, res) => sum + res.confidence, 0) / total;
    
    // Distribuição de classificação (Tipos de Grãos/Defeitos)
    const classCount = results.reduce((acc, res) => {
      acc[res.classification] = (acc[res.classification] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const pieData = Object.entries(classCount).map(([name, value]) => ({
      name,
      value,
      percentage: ((value / total) * 100).toFixed(1)
    }));

    // Distribuição por Qualidade
    const qualityCount = results.reduce((acc, res) => {
      acc[res.details.quality] = (acc[res.details.quality] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const barData = Object.entries(qualityCount).map(([name, value]) => ({
      name,
      value
    }));

    return { total, avgConfidence, pieData, barData };
  }, [results]);

  if (!stats) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-gray-500">
        <PieChartIcon className="w-16 h-16 mb-4 opacity-50" />
        <h2 className="text-xl font-semibold">Nenhum dado disponível</h2>
        <p>Realize um processamento em lote primeiro para visualizar o dashboard.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-white/50 backdrop-blur-sm border-t-4 border-green-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500 flex items-center justify-between">
              Total Analisado
              <LayersIcon className="w-4 h-4 text-green-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.total}</div>
            <p className="text-xs text-gray-500 mt-1">Imagens classificadas</p>
          </CardContent>
        </Card>

        <Card className="bg-white/50 backdrop-blur-sm border-t-4 border-blue-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500 flex items-center justify-between">
              Confiança Média
              <TargetIcon className="w-4 h-4 text-blue-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.avgConfidence.toFixed(1)}%</div>
            <p className="text-xs text-gray-500 mt-1">Nível de certeza da IA</p>
          </CardContent>
        </Card>

        <Card className="bg-white/50 backdrop-blur-sm border-t-4 border-purple-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500 flex items-center justify-between">
              Variedades Identificadas
              <ActivityIcon className="w-4 h-4 text-purple-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.pieData.length}</div>
            <p className="text-xs text-gray-500 mt-1">Categorias distintas</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Composição do Lote</CardTitle>
            <CardDescription>
              Distribuição percentual dos tipos de grãos classificados
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats.pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={80}
                  outerRadius={120}
                  paddingAngle={5}
                  dataKey="value"
                  label={({name, percentage}) => `${name} (${percentage}%)`}
                >
                  {stats.pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value, name, props) => [`${value} grãos (${props.payload.percentage}%)`, name]} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Análise de Qualidade</CardTitle>
            <CardDescription>
              Agrupamento conforme o nível qualitativo da classificação (Excelente, Boa, Regular, Ruim)
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.barData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {stats.barData.map((entry, index) => {
                    const colorMap: Record<string, string> = {
                      'Excelente': '#10b981',
                      'Boa': '#3b82f6',
                      'Regular': '#eab308',
                      'Ruim': '#ef4444'
                    };
                    return <Cell key={`cell-${index}`} fill={colorMap[entry.name] || '#94a3b8'} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Resumo Estatístico</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left border-collapse">
              <thead className="bg-gray-100 text-gray-700 uppercase">
                <tr>
                  <th className="px-6 py-3 border-b">Classificação</th>
                  <th className="px-6 py-3 border-b">Quantidade</th>
                  <th className="px-6 py-3 border-b">Percentual (%)</th>
                </tr>
              </thead>
              <tbody>
                {stats.pieData.map((item, index) => (
                  <tr key={index} className="border-b hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                      {item.name}
                    </td>
                    <td className="px-6 py-4">{item.value}</td>
                    <td className="px-6 py-4">{item.percentage}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
