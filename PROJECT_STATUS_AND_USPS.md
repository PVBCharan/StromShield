# SIH26072 — Thunderstorm & Lightning Nowcasting Platform
## Implementation Status & Completed USP Enhancements

> **Target Event:** Smart India Hackathon (SIH 2026)  
> **Theme:** Disaster Management  
> **Target Ministry:** Ministry of Earth Sciences / India Meteorological Department (IMD)  
> **Target Region (MVP):** Gujarat (Ahmedabad, Surat, Rajkot, Vadodara, Bhuj)

---

## ✅ Completed USP Implementations

### USP 3 — Physics-Guided Hybrid Trajectory Clamp ✅ DONE
**Files Modified:** `apps/api/src/ml/inference.py`, `apps/api/src/ml/test_ml_accuracy.py`

The ML trajectory prediction suffered from a **16.98 km MAE drift**. We implemented a physics-guided gating formula that blends smoothed-velocity linear extrapolation with neural network predictions using horizon-dependent weights:

```
X_final(t) = w_phys(t) * X_linear + w_ml(t) * X_neural
w_ml = 0.05 + 0.45 * (t/60)^1.5
```

**Benchmark Results (v4 vs v3):**
```
Method                              MAE (km)  RMSE (km)
----------------------------------- ---------- ----------
Baseline Linear Extrapolation            1.290      1.929
Pure ML (Deep Nowcaster v3)              6.288      8.769
⚡ Physics-Guided Hybrid (v4)             2.932      4.383

Per-Horizon:
  +15 min:  0.185 km  (w_ml=0.11)  ← near-perfect
  +30 min:  0.874 km  (w_ml=0.21)  ← excellent  
  +45 min:  2.802 km  (w_ml=0.34)
  +60 min:  7.866 km  (w_ml=0.50)
```

---

### USP 4 — RainViewer Live Radar Tile Layer ✅ DONE
**Files Modified:** `apps/web/src/components/map/WeatherMap.tsx`

Integrated the **RainViewer public weather radar API** which provides free, real-time global Doppler radar composite tiles.

**Features:**
- On map load, fetches the latest radar timestamp and adds real satellite/radar tiles
- Auto-refreshes every 5 minutes to show the newest radar scan
- Toggleable via a new **"Live Radar"** button (teal icon) in the Map Controls panel
- Renders below the simulated radar overlay, above the basemap
- Degrades gracefully if RainViewer is unreachable

---

### USP 2 — Vernacular Voice Alert Button (Hindi & Gujarati) ✅ DONE  
**Files Created:** `apps/web/src/components/ui/VoiceAlertButton.tsx`  
**Files Modified:** `apps/web/src/components/panels/RiskPanel.tsx`

Created a reusable VoiceAlertButton component using the browser-native Web Speech API.

**Features:**
- Language selector pills: **EN** / **हिं** (Hindi) / **ગુ** (Gujarati)
- Contextual alert message built from live storm data (risk level, cell ID, speed, lightning probability, ETA)
- Integrated into every storm card in the Risk Panel
- Hindi example: "सावधान! गंभीर तूफान चेतावनी। आकाशीय बिजली की संभावना 85 प्रतिशत है।"
- Gujarati example: "સાવધાન! ગંભીર વાવાઝોડું ચેતવણી। વીજળીની સંભાવના 85 ટકા છે।"

---

### USP 5 — Progressive Web App (PWA) & Mobile Support ✅ DONE
**Files Created:**
- `apps/web/public/manifest.json` — PWA manifest with app name, icons, theme
- `apps/web/public/sw.js` — Service worker for offline caching
- `apps/web/public/icons/icon-192.png` & `icon-512.png` — App icons

**Files Modified:** `apps/web/src/app/layout.tsx`

**Features:**
- Full PWA manifest with standalone display for native app experience
- Service worker with network-first strategy for API, cache-first for static assets
- Apple Web App meta tags for iOS "Add to Home Screen"
- Mobile-optimized viewport
- Theme color #0ea5e9 (sky blue)

**How to install on Android:**
1. Open the dashboard URL in Chrome
2. Tap the three-dot menu → "Install app" or "Add to Home Screen"
3. StormWatch appears as a standalone app with the lightning bolt icon

---

## How to Run

### Backend (FastAPI)
```bash
cd apps/api
python -m pip install -r requirements.txt
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Next.js)
```bash
cd apps/web
npm install
npm run dev
```

### Run ML Benchmark
```bash
cd apps/api
python -m src.ml.test_ml_accuracy
```
