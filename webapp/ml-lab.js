/* ATLAS ML Lab (Lab 14 CCA) — predict playground.
   The trained best regressor from the atlas-ml-cca project, ported to the
   browser coefficient for coefficient:
     pipeline  = StandardScaler -> LinearRegression   (best_regressor.joblib)
     features  = [dist_slope, dist_flat, dist_temperature, dist_relief,
                  dist_soil, soil_missing]
     CV R2 0.9817, CV RMSE 1.003 on the 56 training pairs.
   Preprocessing mirrors backend/main.py /api/predict exactly: when a pair
   has no soil coverage, dist_soil is set to the training median (0.5) and
   soil_missing is set to 1. Inference: z_i = (x_i - mean_i) / scale_i,
   score = intercept + sum(coef_i * z_i), rounded to 1 decimal. */
"use strict";

const ML_MODEL = {
  features: ["dist_slope", "dist_flat", "dist_temperature", "dist_relief", "dist_soil", "soil_missing"],
  mean:  [0.44752272727272724, 0.501090909090909, 0.8466818181818183, 0.49427272727272736, 0.46259090909090916, 0.1590909090909091],
  scale: [0.26221248224885757, 0.3008351117519528, 0.08814986534493731, 0.28938205036660475, 0.2548831566693237, 0.36576083953252464],
  coef:  [-2.807370166100895, -3.1866578674556676, -1.939666272577373, -2.481290842151618, -1.5691020793386439, -0.11787515552203633],
  intercept: 55.068181818181806,
  soilMedian: 0.5,
};

/* inputs: {dist_slope, dist_flat, dist_temperature, dist_relief, dist_soil|null}
   dist_soil null means "no soil coverage", exactly like the API. */
function mlPredict(inp) {
  const soilMissing = (inp.dist_soil === null || inp.dist_soil === undefined) ? 1 : 0;
  const x = [
    inp.dist_slope, inp.dist_flat, inp.dist_temperature, inp.dist_relief,
    soilMissing ? ML_MODEL.soilMedian : inp.dist_soil, soilMissing,
  ];
  let y = ML_MODEL.intercept;
  for (let i = 0; i < x.length; i++) {
    y += ML_MODEL.coef[i] * ((x[i] - ML_MODEL.mean[i]) / ML_MODEL.scale[i]);
  }
  return Math.round(y * 10) / 10;
}
window.mlPredict = mlPredict;

/* ------------------------------------------------------------- playground */
const ML_SLIDERS = [
  ["dist_slope", "Slope distance"],
  ["dist_flat", "Flat-area distance"],
  ["dist_temperature", "Temperature distance"],
  ["dist_relief", "Relief distance"],
  ["dist_soil", "Soil distance"],
];
let mlLabReady = false;

function mlCurrentInputs() {
  const inp = {};
  for (const [key] of ML_SLIDERS) inp[key] = Number(document.getElementById(`ml-${key}`).value);
  if (document.getElementById("mlSoilMissing").checked) inp.dist_soil = null;
  return inp;
}
function mlRenderScore() {
  document.getElementById("mlScore").textContent = mlPredict(mlCurrentInputs()).toFixed(1);
}
function mlSetSliders(vals) {
  for (const [key] of ML_SLIDERS) {
    const el = document.getElementById(`ml-${key}`);
    el.value = vals[key];
    document.getElementById(`mlo-${key}`).textContent = Number(vals[key]).toFixed(2);
  }
  mlRenderScore();
}

function initMlLab() {
  if (mlLabReady) return;
  mlLabReady = true;
  const box = document.getElementById("mlSliders");
  box.innerHTML = ML_SLIDERS.map(([key, label]) => `
    <div class="slider-row">
      <label for="ml-${key}">${label} <output id="mlo-${key}">0.30</output></label>
      <input type="range" id="ml-${key}" min="0" max="1" step="0.01" value="0.3">
    </div>`).join("");
  for (const [key] of ML_SLIDERS) {
    document.getElementById(`ml-${key}`).addEventListener("input", (e) => {
      document.getElementById(`mlo-${key}`).textContent = Number(e.target.value).toFixed(2);
      mlRenderScore();
    });
  }
  const soilBox = document.getElementById("mlSoilMissing");
  const soilSlider = document.getElementById("ml-dist_soil");
  soilBox.addEventListener("change", () => {
    soilSlider.disabled = soilBox.checked;
    soilSlider.closest(".slider-row").style.opacity = soilBox.checked ? "0.45" : "";
    mlRenderScore();
  });
  document.getElementById("mlPresetDemo").addEventListener("click", () => {
    soilBox.checked = false; soilBox.dispatchEvent(new Event("change"));
    mlSetSliders({ dist_slope: 0.3, dist_flat: 0.3, dist_temperature: 0.3, dist_relief: 0.3, dist_soil: 0.3 });
  });
  document.getElementById("mlPresetPair").addEventListener("click", () => {
    soilBox.checked = false; soilBox.dispatchEvent(new Event("change"));
    /* the real Shackleton x Klyuchevskaya pair from data/ml_pairs.csv */
    mlSetSliders({ dist_slope: 0.052, dist_flat: 0.072, dist_temperature: 0.767, dist_relief: 0.21, dist_soil: 0.163 });
  });
  mlRenderScore();
}
window.initMlLab = initMlLab;
