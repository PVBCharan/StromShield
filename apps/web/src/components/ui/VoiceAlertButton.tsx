"use client";

import { useState, useCallback } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { cn } from "@/lib/utils";

interface VoiceAlertButtonProps {
  /** Risk level of the storm */
  riskLevel: string;
  /** Cell ID */
  cellId: string;
  /** ETA in minutes (null if not approaching) */
  etaMinutes?: number | null;
  /** Thunderstorm probability 0-1 */
  thunderstormProb?: number;
  /** Lightning probability 0-1 */
  lightningProb?: number;
  /** Speed km/h */
  speedKmh?: number;
  /** Optional class name */
  className?: string;
}

type Language = "en" | "hi" | "gu";

const LANG_LABELS: Record<Language, string> = {
  en: "EN",
  hi: "हिं",
  gu: "ગુ",
};

const LANG_NAMES: Record<Language, string> = {
  en: "English",
  hi: "Hindi",
  gu: "Gujarati",
};

function buildAlertMessage(
  lang: Language,
  riskLevel: string,
  cellId: string,
  etaMinutes: number | null | undefined,
  thunderstormProb: number,
  lightningProb: number,
  speedKmh: number,
): string {
  const tsPct = Math.round(thunderstormProb * 100);
  const ltPct = Math.round(lightningProb * 100);
  const etaStr = etaMinutes ? `${Math.round(etaMinutes)}` : null;
  const level = riskLevel.toUpperCase();

  if (lang === "hi") {
    const severity =
      riskLevel === "severe" ? "अत्यंत गंभीर" :
      riskLevel === "high" ? "गंभीर" :
      riskLevel === "moderate" ? "मध्यम" : "सामान्य";

    let msg = `सावधान! ${severity} तूफान चेतावनी। स्टॉर्म सेल ${cellId} ${Math.round(speedKmh)} किलोमीटर प्रति घंटे की रफ्तार से आ रहा है।`;
    msg += ` आकाशीय बिजली की संभावना ${ltPct} प्रतिशत है।`;
    msg += ` तूफान की संभावना ${tsPct} प्रतिशत है।`;
    if (etaStr) {
      msg += ` अहमदाबाद तक पहुंचने का अनुमानित समय ${etaStr} मिनट है।`;
    }
    msg += ` कृपया सुरक्षित स्थान पर जाएं। खुले मैदान, पेड़ों और बिजली के खंभों से दूर रहें।`;
    return msg;
  }

  if (lang === "gu") {
    const severity =
      riskLevel === "severe" ? "અત્યંત ગંભીર" :
      riskLevel === "high" ? "ગંભીર" :
      riskLevel === "moderate" ? "મધ્યમ" : "સામાન્ય";

    let msg = `સાવધાન! ${severity} વાવાઝોડું ચેતવણી। સ્ટોર્મ સેલ ${cellId} ${Math.round(speedKmh)} કિલોમીટર પ્રતિ કલાકની ઝડપે આવી રહ્યું છે।`;
    msg += ` વીજળીની સંભાવના ${ltPct} ટકા છે।`;
    msg += ` વાવાઝોડાની સંભાવના ${tsPct} ટકા છે।`;
    if (etaStr) {
      msg += ` અમદાવાદ સુધી પહોંચવાનો અંદાજિત સમય ${etaStr} મિનિટ છે।`;
    }
    msg += ` કૃપા કરીને સુરક્ષિત સ્થળે આશ્રય લો। ખુલ્લા મેદાન, ઝાડ અને વીજળીના થાંભલાથી દૂર રહો।`;
    return msg;
  }

  // English
  let msg = `Alert! ${level} storm warning. Storm cell ${cellId} is approaching at ${Math.round(speedKmh)} kilometers per hour.`;
  msg += ` Lightning probability is ${ltPct} percent.`;
  msg += ` Thunderstorm probability is ${tsPct} percent.`;
  if (etaStr) {
    msg += ` Estimated time of arrival at Ahmedabad is ${etaStr} minutes.`;
  }
  msg += ` Please seek shelter immediately. Stay away from open fields, isolated trees, and power lines.`;
  return msg;
}

function getVoiceLang(lang: Language): string {
  if (lang === "hi") return "hi-IN";
  if (lang === "gu") return "gu-IN";
  return "en-IN";
}

export function VoiceAlertButton({
  riskLevel,
  cellId,
  etaMinutes,
  thunderstormProb = 0.5,
  lightningProb = 0.3,
  speedKmh = 20,
  className,
}: VoiceAlertButtonProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [selectedLang, setSelectedLang] = useState<Language>("hi");

  const speak = useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    // Stop any current speech
    window.speechSynthesis.cancel();

    if (isSpeaking) {
      setIsSpeaking(false);
      return;
    }

    const message = buildAlertMessage(
      selectedLang,
      riskLevel,
      cellId,
      etaMinutes,
      thunderstormProb,
      lightningProb,
      speedKmh,
    );

    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = getVoiceLang(selectedLang);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Try to find a matching voice
    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(v => v.lang.startsWith(getVoiceLang(selectedLang).split("-")[0]));
    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, [selectedLang, riskLevel, cellId, etaMinutes, thunderstormProb, lightningProb, speedKmh, isSpeaking]);

  return (
    <div className={cn("flex items-center gap-1.5", className)}>
      {/* Language selector pills */}
      <div className="flex gap-0.5 rounded-md bg-black/30 p-0.5 border border-white/10">
        {(["en", "hi", "gu"] as Language[]).map((lang) => (
          <button
            key={lang}
            type="button"
            onClick={() => { setSelectedLang(lang); window.speechSynthesis?.cancel(); setIsSpeaking(false); }}
            title={LANG_NAMES[lang]}
            className={cn(
              "rounded px-1.5 py-0.5 text-[9px] font-bold transition-all",
              selectedLang === lang
                ? "bg-amber-500/30 text-amber-200 border border-amber-500/40"
                : "text-slate-500 hover:text-slate-300"
            )}
          >
            {LANG_LABELS[lang]}
          </button>
        ))}
      </div>

      {/* Speak / Stop button */}
      <button
        type="button"
        onClick={speak}
        title={isSpeaking ? "Stop alert" : `Speak alert in ${LANG_NAMES[selectedLang]}`}
        className={cn(
          "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-all border",
          isSpeaking
            ? "bg-red-500/20 text-red-300 border-red-500/40 animate-pulse"
            : "bg-amber-500/15 text-amber-300 border-amber-500/30 hover:bg-amber-500/25 hover:text-amber-200"
        )}
      >
        {isSpeaking ? (
          <>
            <VolumeX className="h-3.5 w-3.5" />
            <span>Stop</span>
          </>
        ) : (
          <>
            <Volume2 className="h-3.5 w-3.5" />
            <span>🔊 Voice Alert</span>
          </>
        )}
      </button>
    </div>
  );
}
