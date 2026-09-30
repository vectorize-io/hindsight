"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, Film, X, AlertCircle } from "lucide-react";
import { cn, formatFileSize } from "@/lib/utils";

interface VideoUploaderProps {
  onFileSelected: (file: File) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

const ACCEPTED = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/mpeg", "video/3gpp"];
const MAX_MB = 200;

export function VideoUploader({ onFileSelected, isLoading, disabled }: VideoUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    setError(null);
    if (!file.type.startsWith("video/")) {
      setError("Please upload a video file (MP4, MOV, WebM, etc.)");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File too large. Maximum size is ${MAX_MB}MB.`);
      return;
    }
    setSelectedFile(file);
    onFileSelected(file);
  }, [onFileSelected]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full">
      {selectedFile ? (
        <div className="glass rounded-2xl p-5 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-brand-500/20 flex items-center justify-center flex-shrink-0">
            <Film className="w-6 h-6 text-brand-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">{selectedFile.name}</p>
            <p className="text-xs text-gray-500 mt-0.5">{formatFileSize(selectedFile.size)}</p>
          </div>
          {!isLoading && (
            <button onClick={clearFile} className="text-gray-500 hover:text-gray-300 transition-colors flex-shrink-0">
              <X className="w-4 h-4" />
            </button>
          )}
          {isLoading && (
            <div className="w-4 h-4 rounded-full border-2 border-brand-500 border-t-transparent animate-spin flex-shrink-0" />
          )}
        </div>
      ) : (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !disabled && inputRef.current?.click()}
          className={cn(
            "relative rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-200",
            dragOver
              ? "border-brand-400 bg-brand-500/10"
              : "border-white/10 hover:border-white/25 hover:bg-white/[0.02]",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            onChange={onInputChange}
            className="hidden"
            disabled={disabled}
          />

          <div className="flex flex-col items-center gap-3">
            <div className={cn(
              "w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-200",
              dragOver ? "bg-brand-500/20" : "bg-white/[0.04]"
            )}>
              <Upload className={cn("w-7 h-7 transition-colors", dragOver ? "text-brand-400" : "text-gray-500")} />
            </div>

            <div>
              <p className="text-white font-semibold mb-1">
                {dragOver ? "Drop your video here" : "Upload your Reel"}
              </p>
              <p className="text-sm text-gray-500">
                Drag & drop or click to browse
              </p>
              <p className="text-xs text-gray-600 mt-1">
                MP4, MOV, WebM, AVI — up to {MAX_MB}MB
              </p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 mt-3 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
