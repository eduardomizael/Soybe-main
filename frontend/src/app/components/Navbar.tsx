import { LayoutDashboardIcon, CameraIcon, ActivityIcon } from "lucide-react";
import { Button } from "./ui/button";

interface NavbarProps {
  activeTab: "classifier" | "dashboard" | "training";
  onTabChange: (tab: "classifier" | "dashboard" | "training") => void;
  hasResults?: boolean;
}

export function Navbar({ activeTab, onTabChange, hasResults = false }: NavbarProps) {
  return (
    <nav className="bg-white border-b sticky top-0 z-50 shadow-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center gap-2">
            <img src="/soyd-logo.png" alt="Logo" className="h-8 w-8 object-contain hidden" />
            <span className="font-bold text-xl text-green-800">Soybe System</span>
          </div>
          
          <div className="flex gap-4">
            <Button
              variant={activeTab === "classifier" ? "default" : "ghost"}
              onClick={() => onTabChange("classifier")}
              className="gap-2"
            >
              <CameraIcon className="w-4 h-4" />
              Classificador
            </Button>
            
            <Button
              variant={activeTab === "training" ? "default" : "ghost"}
              onClick={() => onTabChange("training")}
              className="gap-2"
            >
              <ActivityIcon className="w-4 h-4" />
              Treinamento
            </Button>

            <Button
              variant={activeTab === "dashboard" ? "default" : "ghost"}
              onClick={() => onTabChange("dashboard")}
              disabled={!hasResults && activeTab !== "dashboard"}
              className="gap-2"
            >
              <LayoutDashboardIcon className="w-4 h-4" />
              Dashboard
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}
