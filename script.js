const CANVAS_SIZE = 280;
const MODEL_MANIFEST_URL = "model/manifest.json";
const MODEL_WEIGHTS_URL = "model/weights.bin";

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const canvasHint = document.getElementById("canvasHint");
const btnClear = document.getElementById("btnClear");
const btnUndo = document.getElementById("btnUndo");
const brushSizeInput = document.getElementById("brushSize");
const brushSizeLabel = document.getElementById("brushSizeLabel");
const statusBadge = document.getElementById("statusBadge");
const emptyState = document.getElementById("emptyState");
const topResult = document.getElementById("topResult");
const probBars = document.getElementById("probBars");
const confusionPanel = document.getElementById("confusionPanel");
const confusionContent = document.getElementById("confusionContent");
const referenceGrid = document.getElementById("referenceGrid");
const themeToggle = document.getElementById("themeToggle");

let model = null;
let modelLoadPromise = null;
let refClassesCache = [];
let isDrawing = false;
let hasDrawn = false;
let lastX = 0;
let lastY = 0;
let brushSize = 6;
let activePointerType = null;
let canvasStates = [];
let clearCanvasTimeout = null;
let isLightMode = localStorage.getItem("theme") === "light";
const MAX_UNDO = 30;

const EQUIVALENT_GROUPS = [
    new Set(["greek_15_omicron", "english_o"]),
];

const CONFUSION_PAIRS = [
    ["greek_13_nu", "english_v"],
    ["greek_17_rho", "english_p"],
    ["greek_01_alpha", "english_a"],
    ["greek_22_chi", "english_x"],
    ["greek_15_omicron", "english_o"],
    ["greek_10_kappa", "english_k"],
    ["greek_19_tau", "english_t"],
    ["greek_20_upsilon", "english_u"],
    ["greek_07_eta", "english_n"],
    ["greek_09_iota", "english_i"],
    ["greek_09_iota", "english_l"],
    ["greek_24_omega", "english_w"],
    ["greek_18_sigma", "english_s"],
];

const CONFUSION_MAP = new Map();
for (const [a, b] of CONFUSION_PAIRS) {
    if (!CONFUSION_MAP.has(a)) CONFUSION_MAP.set(a, []);
    if (!CONFUSION_MAP.has(b)) CONFUSION_MAP.set(b, []);
    CONFUSION_MAP.get(a).push(b);
    CONFUSION_MAP.get(b).push(a);
}

function setStatus(kind, text) {
    const dot = statusBadge.querySelector(".status-dot");
    const label = statusBadge.querySelector(".status-text");
    dot.className = kind ? `status-dot ${kind}` : "status-dot";
    label.textContent = text;
}

function getDisplaySymbol(entry) {
    return entry?.display_symbol || entry?.symbol || "";
}

function getCleanClassName(name) {
    return (name || "")
        .replace("greek_", "")
        .replace("english_", "")
        .replace(/^\d+_/, "")
        .replaceAll("_", " ");
}

function isDualSymbol(symbol) {
    return typeof symbol === "string" && (symbol.includes("/") || symbol.includes("(") || symbol.length > 3);
}

function getEquivalentGroup(className) {
    for (const group of EQUIVALENT_GROUPS) {
        if (group.has(className)) return group;
    }
    return null;
}

function updateThemeToggleLabel() {
    if (!themeToggle) return;
    themeToggle.textContent = isLightMode ? "切換深色" : "切換淺色";
    themeToggle.setAttribute("aria-label", isLightMode ? "切換深色模式" : "切換淺色模式");
}

function initCanvas() {
    ctx.fillStyle = isLightMode ? "#ffffff" : "#0d1117";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.strokeStyle = isLightMode ? "#1e293b" : "#e2e8f0";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;
}

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY,
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function stampBrush(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, Math.max(brushSize / 2, 1), 0, Math.PI * 2);
    ctx.fillStyle = isLightMode ? "#1e293b" : "#e2e8f0";
    ctx.fill();
}

function drawContinuousStroke(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const distance = Math.hypot(dx, dy);

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();

    const step = Math.max(1, brushSize * 0.35);
    const steps = Math.max(1, Math.ceil(distance / step));
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        stampBrush(p1.x + dx * t, p1.y + dy * t);
    }
}

function startDraw(e) {
    if (e.type === "mousedown" && e.button !== 0) return;
    e.preventDefault();
    canvasStates.push(ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE));
    if (canvasStates.length > MAX_UNDO) canvasStates.shift();
    isDrawing = true;
    hasDrawn = true;
    activePointerType = e.touches ? "touch" : "mouse";
    canvasHint.classList.add("hidden");
    canvas.classList.add("drawing");

    if (clearCanvasTimeout) {
        clearTimeout(clearCanvasTimeout);
        clearCanvasTimeout = null;
        [topResult, probBars, confusionPanel].forEach(el => {
            el.style.transition = "";
            el.style.opacity = "";
        });
    }

    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;
    stampBrush(pos.x, pos.y);
}

function draw(e) {
    if (!isDrawing) return;
    if (activePointerType === "mouse" && e.buttons !== undefined && (e.buttons & 1) === 0) {
        endDraw();
        return;
    }
    e.preventDefault();
    const pos = getPos(e);
    ctx.strokeStyle = isLightMode ? "#1e293b" : "#e2e8f0";
    ctx.lineWidth = brushSize;
    drawContinuousStroke({ x: lastX, y: lastY }, pos);
    lastX = pos.x;
    lastY = pos.y;
}

function endDraw(e) {
    if (e) e.preventDefault();
    if (!isDrawing) return;
    isDrawing = false;
    activePointerType = null;
    canvas.classList.remove("drawing");
    if (hasDrawn) recognize();
}

function clearCanvas() {
    [topResult, probBars, confusionPanel].forEach(el => {
        el.style.transition = "opacity 0.5s ease";
        el.style.opacity = "0";
    });
    document.getElementById("canvasWrapper").classList.remove("confidence-high", "confidence-medium", "confidence-low");
    canvasStates = [];
    document.querySelector(".border-flow-svg").setAttribute("class", "border-flow-svg");
    initCanvas();
    hasDrawn = false;
    isDrawing = false;
    activePointerType = null;
    canvasHint.classList.remove("hidden");

    if (clearCanvasTimeout) clearTimeout(clearCanvasTimeout);
    clearCanvasTimeout = setTimeout(() => {
        emptyState.classList.remove("hidden");
        topResult.classList.add("hidden");
        probBars.classList.add("hidden");
        confusionPanel.classList.add("hidden");
        [topResult, probBars, confusionPanel].forEach(el => {
            el.style.transition = "";
            el.style.opacity = "";
        });
        clearCanvasTimeout = null;
    }, 500);
}

function undoLastStroke() {
    if (canvasStates.length === 0) return;
    const state = canvasStates.pop();
    ctx.putImageData(state, 0, 0);
    hasDrawn = canvasStates.length > 0;
    if (!hasDrawn) {
        canvasHint.classList.remove("hidden");
        emptyState.classList.remove("hidden");
        topResult.classList.add("hidden");
        probBars.classList.add("hidden");
        confusionPanel.classList.add("hidden");
        document.getElementById("canvasWrapper").classList.remove("confidence-high", "confidence-medium", "confidence-low");
        document.querySelector(".border-flow-svg").setAttribute("class", "border-flow-svg");
    } else {
        recognize();
    }
}

function handleGlobalKeydown(e) {
    if (e.ctrlKey || e.metaKey) {
        if (e.key === "z" || e.key === "Z") {
            e.preventDefault();
            undoLastStroke();
        }
        return;
    }
    if (e.altKey || e.isComposing || e.repeat) return;
    if (e.key === "d" || e.key === "D") {
        e.preventDefault();
        clearCanvas();
    }
}

async function loadBrowserModel() {
    if (model) return model;
    if (modelLoadPromise) return modelLoadPromise;

    modelLoadPromise = (async () => {
        setStatus("", "載入模型中...");
        const [manifestRes, weightsRes] = await Promise.all([
            fetch(MODEL_MANIFEST_URL),
            fetch(MODEL_WEIGHTS_URL),
        ]);
        if (!manifestRes.ok || !weightsRes.ok) {
            throw new Error("無法載入模型檔，請確認是透過 GitHub Pages 或本機伺服器開啟。");
        }
        const manifest = await manifestRes.json();
        const buffer = await weightsRes.arrayBuffer();
        const tensors = {};
        for (const [name, meta] of Object.entries(manifest.tensors)) {
            tensors[name] = {
                data: new Float32Array(buffer, meta.offset, meta.length),
                shape: meta.shape,
            };
        }
        model = { tensors, classes: manifest.classes, modelInfo: manifest.model_info };
        refClassesCache = manifest.classes;
        renderReferenceGrid();
        setStatus("online", "模型就緒");
        return model;
    })().catch(error => {
        setStatus("offline", "模型載入失敗");
        showError(error.message);
        throw error;
    });

    return modelLoadPromise;
}

function preprocessCanvas() {
    const image = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;
    let arr = new Float32Array(CANVAS_SIZE * CANVAS_SIZE);
    let sum = 0;
    for (let i = 0, p = 0; i < image.length; i += 4, p++) {
        const gray = image[i] * 0.299 + image[i + 1] * 0.587 + image[i + 2] * 0.114;
        arr[p] = gray;
        sum += gray;
    }
    if (sum / arr.length > 127) {
        for (let i = 0; i < arr.length; i++) arr[i] = 255 - arr[i];
    }

    let maxVal = 0;
    for (const value of arr) if (value > maxVal) maxVal = value;
    const threshold = Math.max(maxVal * 0.15, 10);
    let rMin = CANVAS_SIZE, rMax = -1, cMin = CANVAS_SIZE, cMax = -1;
    for (let y = 0; y < CANVAS_SIZE; y++) {
        for (let x = 0; x < CANVAS_SIZE; x++) {
            if (arr[y * CANVAS_SIZE + x] > threshold) {
                if (y < rMin) rMin = y;
                if (y > rMax) rMax = y;
                if (x < cMin) cMin = x;
                if (x > cMax) cMax = x;
            }
        }
    }

    if (rMax < 0) {
        return new Float32Array(28 * 28).fill(-1);
    }

    rMin = Math.max(0, rMin - 2);
    rMax = Math.min(CANVAS_SIZE - 1, rMax + 2);
    cMin = Math.max(0, cMin - 2);
    cMax = Math.min(CANVAS_SIZE - 1, cMax + 2);

    const cropW = cMax - cMin + 1;
    const cropH = rMax - rMin + 1;
    const target = 24;
    const scale = Math.min(target / Math.max(cropW, 1), target / Math.max(cropH, 1));
    const newW = Math.max(1, Math.floor(cropW * scale));
    const newH = Math.max(1, Math.floor(cropH * scale));

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext("2d");
    const cropImage = cropCtx.createImageData(cropW, cropH);
    for (let y = 0; y < cropH; y++) {
        for (let x = 0; x < cropW; x++) {
            const value = arr[(rMin + y) * CANVAS_SIZE + (cMin + x)];
            const idx = (y * cropW + x) * 4;
            cropImage.data[idx] = value;
            cropImage.data[idx + 1] = value;
            cropImage.data[idx + 2] = value;
            cropImage.data[idx + 3] = 255;
        }
    }
    cropCtx.putImageData(cropImage, 0, 0);

    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = newW;
    resizeCanvas.height = newH;
    const resizeCtx = resizeCanvas.getContext("2d");
    resizeCtx.imageSmoothingEnabled = true;
    resizeCtx.imageSmoothingQuality = "high";
    resizeCtx.drawImage(cropCanvas, 0, 0, newW, newH);
    const resized = resizeCtx.getImageData(0, 0, newW, newH).data;

    const final = new Float32Array(28 * 28);
    const xOff = Math.floor((28 - newW) / 2);
    const yOff = Math.floor((28 - newH) / 2);
    for (let y = 0; y < newH; y++) {
        for (let x = 0; x < newW; x++) {
            final[(yOff + y) * 28 + (xOff + x)] = resized[(y * newW + x) * 4];
        }
    }

    let mx = 0;
    for (let i = 0; i < final.length; i++) {
        if (final[i] < 80) final[i] = 0;
        if (final[i] > mx) mx = final[i];
    }
    const tensor = new Float32Array(28 * 28);
    for (let i = 0; i < final.length; i++) {
        const normalized = mx > 0 ? (final[i] / mx) : 0;
        tensor[i] = normalized * 2 - 1;
    }
    return tensor;
}

function conv2d(input, inC, inH, inW, weight, bias, outC) {
    const out = new Float32Array(outC * inH * inW);
    const kernel = 3;
    for (let oc = 0; oc < outC; oc++) {
        const biasValue = bias[oc];
        for (let y = 0; y < inH; y++) {
            for (let x = 0; x < inW; x++) {
                let sum = biasValue;
                for (let ic = 0; ic < inC; ic++) {
                    const inputBase = ic * inH * inW;
                    const weightBase = ((oc * inC + ic) * kernel) * kernel;
                    for (let ky = 0; ky < kernel; ky++) {
                        const iy = y + ky - 1;
                        if (iy < 0 || iy >= inH) continue;
                        for (let kx = 0; kx < kernel; kx++) {
                            const ix = x + kx - 1;
                            if (ix < 0 || ix >= inW) continue;
                            sum += input[inputBase + iy * inW + ix] * weight[weightBase + ky * kernel + kx];
                        }
                    }
                }
                out[oc * inH * inW + y * inW + x] = sum;
            }
        }
    }
    return out;
}

function batchNormRelu(input, channels, height, width, gamma, beta, mean, variance) {
    const size = height * width;
    for (let c = 0; c < channels; c++) {
        const scale = gamma[c] / Math.sqrt(variance[c] + 1e-5);
        const offset = beta[c] - mean[c] * scale;
        const base = c * size;
        for (let i = 0; i < size; i++) {
            const value = input[base + i] * scale + offset;
            input[base + i] = value > 0 ? value : 0;
        }
    }
    return input;
}

function maxPool2(input, channels, inH, inW) {
    const outH = Math.floor(inH / 2);
    const outW = Math.floor(inW / 2);
    const out = new Float32Array(channels * outH * outW);
    for (let c = 0; c < channels; c++) {
        const inBase = c * inH * inW;
        const outBase = c * outH * outW;
        for (let y = 0; y < outH; y++) {
            for (let x = 0; x < outW; x++) {
                const i0 = inBase + (y * 2) * inW + x * 2;
                out[outBase + y * outW + x] = Math.max(
                    input[i0],
                    input[i0 + 1],
                    input[i0 + inW],
                    input[i0 + inW + 1],
                );
            }
        }
    }
    return out;
}

function adaptiveAvgPool3(input, channels, inH, inW) {
    const outH = 3;
    const outW = 3;
    const out = new Float32Array(channels * outH * outW);
    for (let c = 0; c < channels; c++) {
        const base = c * inH * inW;
        for (let oy = 0; oy < outH; oy++) {
            const yStart = Math.floor(oy * inH / outH);
            const yEnd = Math.ceil((oy + 1) * inH / outH);
            for (let ox = 0; ox < outW; ox++) {
                const xStart = Math.floor(ox * inW / outW);
                const xEnd = Math.ceil((ox + 1) * inW / outW);
                let sum = 0;
                let count = 0;
                for (let y = yStart; y < yEnd; y++) {
                    for (let x = xStart; x < xEnd; x++) {
                        sum += input[base + y * inW + x];
                        count++;
                    }
                }
                out[c * outH * outW + oy * outW + ox] = sum / count;
            }
        }
    }
    return out;
}

function linear(input, weight, bias, outFeatures) {
    const inFeatures = input.length;
    const out = new Float32Array(outFeatures);
    for (let o = 0; o < outFeatures; o++) {
        let sum = bias[o];
        const weightBase = o * inFeatures;
        for (let i = 0; i < inFeatures; i++) {
            sum += input[i] * weight[weightBase + i];
        }
        out[o] = sum;
    }
    return out;
}

function relu(input) {
    for (let i = 0; i < input.length; i++) {
        if (input[i] < 0) input[i] = 0;
    }
    return input;
}

function softmax(logits) {
    let maxValue = -Infinity;
    for (const value of logits) if (value > maxValue) maxValue = value;
    let sum = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        probs[i] = Math.exp(logits[i] - maxValue);
        sum += probs[i];
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    return probs;
}

function t(name) {
    return model.tensors[name].data;
}

function runModel(input) {
    let x = input;
    x = conv2d(x, 1, 28, 28, t("features.0.weight"), t("features.0.bias"), 32);
    x = batchNormRelu(x, 32, 28, 28, t("features.1.weight"), t("features.1.bias"), t("features.1.running_mean"), t("features.1.running_var"));
    x = maxPool2(x, 32, 28, 28);
    x = conv2d(x, 32, 14, 14, t("features.4.weight"), t("features.4.bias"), 64);
    x = batchNormRelu(x, 64, 14, 14, t("features.5.weight"), t("features.5.bias"), t("features.5.running_mean"), t("features.5.running_var"));
    x = maxPool2(x, 64, 14, 14);
    x = conv2d(x, 64, 7, 7, t("features.8.weight"), t("features.8.bias"), 128);
    x = batchNormRelu(x, 128, 7, 7, t("features.9.weight"), t("features.9.bias"), t("features.9.running_mean"), t("features.9.running_var"));
    x = adaptiveAvgPool3(x, 128, 7, 7);
    x = linear(x, t("classifier.1.weight"), t("classifier.1.bias"), 256);
    x = relu(x);
    x = linear(x, t("classifier.4.weight"), t("classifier.4.bias"), 50);
    return softmax(x);
}

function buildPredictionResult(probs) {
    const indices = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
    const predictions = indices.map((idx, rankIndex) => {
        const klass = model.classes[idx];
        return {
            rank: rankIndex + 1,
            class_name: klass.name,
            symbol: klass.symbol,
            display_symbol: klass.display_symbol,
            language: klass.language,
            probability: probs[idx],
        };
    });

    const topClass = predictions[0].class_name;
    const confusing = (CONFUSION_MAP.get(topClass) || []).map(className => {
        const idx = model.classes.findIndex(item => item.name === className);
        const klass = model.classes[idx];
        return {
            class_name: klass.name,
            symbol: klass.symbol,
            display_symbol: klass.display_symbol,
            language: klass.language,
            probability: probs[idx],
        };
    });

    return { predictions, confusing };
}

async function recognize() {
    if (!hasDrawn) return;
    try {
        await loadBrowserModel();
        const input = preprocessCanvas();
        const probs = runModel(input);
        renderResults(buildPredictionResult(probs));
    } catch (error) {
        console.error("辨識失敗:", error);
        showError(error.message);
    }
}

function mergeEquivalentPredictions(predictions) {
    const result = [];
    const used = new Set();
    const predMap = {};
    predictions.forEach((p, i) => { predMap[p.class_name] = { pred: p, idx: i }; });

    for (let i = 0; i < predictions.length; i++) {
        if (used.has(i)) continue;
        const group = getEquivalentGroup(predictions[i].class_name);
        if (!group) {
            result.push({ ...predictions[i], merged: false });
            continue;
        }

        const members = [];
        for (const className of group) {
            if (predMap[className] && !used.has(predMap[className].idx)) {
                members.push(predMap[className].pred);
                used.add(predMap[className].idx);
            }
        }

        if (members.length > 0) {
            const symbols = [];
            const names = [];
            const langs = new Set();
            let totalProb = 0;
            for (const className of group) {
                const member = members.find(item => item.class_name === className);
                const ref = refClassesCache.find(item => item.name === className);
                const source = member || ref;
                if (source) {
                    symbols.push(getDisplaySymbol(source).replace(/\s*\(.+?\)\s*/g, ""));
                    names.push(getCleanClassName(source.name || source.class_name));
                    langs.add(source.language);
                    if (member) totalProb += member.probability;
                }
            }
            const uniqueSymbols = [...new Set(symbols)];
            result.push({
                class_name: [...group].join(" / "),
                display_symbol: uniqueSymbols.join(" / "),
                symbol: uniqueSymbols.join(" / "),
                language: [...langs].join("+"),
                probability: Math.min(1, totalProb),
                merged: true,
                original_languages: [...langs],
            });
        }
    }

    result.sort((a, b) => b.probability - a.probability);
    return result;
}

function renderResults(data) {
    const { predictions, confusing } = data;
    if (!predictions || predictions.length === 0) {
        showError("未取得預測結果");
        return;
    }

    const mergedPredictions = mergeEquivalentPredictions(predictions);
    emptyState.classList.add("hidden");
    [topResult, probBars, confusionPanel].forEach(el => {
        el.style.transition = "";
        el.style.opacity = "";
    });

    const topProb = mergedPredictions[0].probability;
    const canvasWrapper = document.getElementById("canvasWrapper");
    canvasWrapper.classList.remove("confidence-high", "confidence-medium", "confidence-low");
    let glowClass = "";
    if (topProb >= 0.9) {
        canvasWrapper.classList.add("confidence-high");
        glowClass = "glow-high";
    } else if (topProb >= 0.7) {
        canvasWrapper.classList.add("confidence-medium");
        glowClass = "glow-medium";
    } else {
        canvasWrapper.classList.add("confidence-low");
        glowClass = "glow-low";
    }

    const glowSvg = document.querySelector(".border-flow-svg");
    const glowRect = document.querySelector(".border-flow-rect");
    glowSvg.setAttribute("class", "border-flow-svg");
    glowRect.getAnimations().forEach(a => a.cancel());
    const colors = { "glow-high": "#22c55e", "glow-medium": "#f59e0b", "glow-low": "#ef4444" };
    if (glowClass && colors[glowClass]) {
        glowSvg.classList.add(glowClass);
        glowRect.setAttribute("stroke", colors[glowClass]);
        glowRect.animate([
            { strokeDashoffset: 0, opacity: 0 },
            { strokeDashoffset: 0, opacity: 0.7, offset: 0.08 },
            { strokeDashoffset: -1260, opacity: 0.7, offset: 0.85 },
            { strokeDashoffset: -1260, opacity: 0, offset: 1 }
        ], { duration: 2000, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" });
    }

    const top = mergedPredictions[0];
    const topSymbolEl = document.getElementById("topSymbol");
    const topDisplaySymbol = getDisplaySymbol(top);
    topSymbolEl.textContent = topDisplaySymbol;
    topSymbolEl.className = `top-result-symbol${isDualSymbol(topDisplaySymbol) ? " dual-symbol" : ""} ${top.merged ? "" : top.class_name}`;
    document.getElementById("topName").textContent = getCleanClassName(top.class_name);
    const langEl = document.getElementById("topLang");
    if (top.merged) {
        langEl.textContent = top.original_languages.map(l => l === "greek" ? "希臘字母" : "英文字母").join(" / ");
        langEl.className = "top-result-lang";
    } else {
        langEl.textContent = top.language === "greek" ? "希臘字母" : "英文字母";
        langEl.className = `top-result-lang ${top.language}`;
    }
    document.getElementById("topProb").textContent = (top.probability * 100).toFixed(1) + "%";
    topResult.classList.remove("hidden");

    probBars.innerHTML = "";
    const maxProb = mergedPredictions[0].probability;
    mergedPredictions.forEach((pred) => {
        const pct = (pred.probability * 100).toFixed(1);
        const barWidth = Math.max(2, (pred.probability / Math.max(maxProb, 0.01)) * 100);
        const displaySymbol = getDisplaySymbol(pred);
        const langLabel = pred.merged
            ? pred.original_languages.map(l => l === "greek" ? "希臘" : "英文").join("+")
            : (pred.language === "greek" ? "希臘" : "英文");
        const fillClass = pred.merged ? "merged" : pred.language;
        const item = document.createElement("div");
        item.className = "prob-bar-item";
        item.innerHTML = `
            <span class="prob-bar-label ${isDualSymbol(displaySymbol) ? "dual-symbol" : ""} ${pred.merged ? "" : pred.class_name}">${displaySymbol}<small class="prob-bar-lang">(${langLabel})</small></span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${fillClass}" style="width: 0%"></div>
            </div>
            <span class="prob-bar-value">${pct}%</span>
        `;
        probBars.appendChild(item);
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                item.querySelector(".prob-bar-fill").style.width = barWidth + "%";
            });
        });
    });
    probBars.classList.remove("hidden");

    const topGroup = getEquivalentGroup(predictions[0].class_name);
    const filteredConfusing = (confusing || []).filter(c => !topGroup || !topGroup.has(c.class_name));
    if (filteredConfusing.length > 0) {
        confusionContent.innerHTML = "";
        filteredConfusing.forEach((c) => {
            const lang = c.language === "greek" ? "希臘" : "英文";
            const displaySymbol = getDisplaySymbol(c);
            const pct = c.probability != null ? (c.probability * 100).toFixed(1) + "%" : "";
            const div = document.createElement("div");
            div.className = "confusion-item";
            div.innerHTML = `
                <span class="confusion-symbol ${isDualSymbol(displaySymbol) ? "dual-symbol" : ""} ${c.class_name}">${displaySymbol}</span>
                <span class="confusion-text">
                    易與 <strong>${displaySymbol}</strong> (${lang} ${getCleanClassName(c.class_name)}) 混淆
                    ${pct ? `<span class="confusion-prob">機率 ${pct}</span>` : ""}
                </span>
            `;
            confusionContent.appendChild(div);
        });
        confusionPanel.classList.remove("hidden");
    } else {
        confusionContent.innerHTML = "";
        confusionPanel.classList.add("hidden");
    }
}

function showError(msg) {
    emptyState.classList.remove("hidden");
    emptyState.innerHTML = `
        <div class="empty-icon">!</div>
        <p style="color: var(--danger);">${msg}</p>
        <p style="margin-top:8px;font-size:12px;color:var(--text-muted);">
            靜態版需要透過 GitHub Pages 或本機靜態伺服器開啟，直接點 file:// 可能無法載入模型檔。
        </p>
    `;
    topResult.classList.add("hidden");
    probBars.classList.add("hidden");
    confusionPanel.classList.add("hidden");
}

function createLetterItem(letter) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "ref-item ref-button";
    const cleanName = getCleanClassName(letter.name);
    const rawSymbol = getDisplaySymbol(letter);
    const displaySymbol = rawSymbol.replace(/\s*\(.+?\)\s*/g, "");
    const symbolClass = isDualSymbol(displaySymbol) ? "ref-symbol dual-symbol" : "ref-symbol";
    item.innerHTML = `
        <span class="${symbolClass} ${letter.name}" style="color: var(--${letter.language}-color, var(--text-primary))">${displaySymbol}</span>
        <span class="ref-name">${cleanName}</span>
    `;
    item.title = `${letter.language === "greek" ? "希臘" : "英文"} ${cleanName}`;
    return item;
}

function renderReferenceGrid() {
    if (!referenceGrid || !refClassesCache.length) return;
    referenceGrid.innerHTML = "";
    const greekLetters = refClassesCache.filter(l => l.language === "greek");
    const englishLetters = refClassesCache.filter(l => l.language === "english");

    const greekCol = document.createElement("div");
    greekCol.className = "ref-column";
    greekCol.innerHTML = '<div class="ref-column-title">希臘字母</div>';
    const greekItems = document.createElement("div");
    greekItems.className = "ref-column-items";
    greekLetters.forEach(l => greekItems.appendChild(createLetterItem(l)));
    greekCol.appendChild(greekItems);
    referenceGrid.appendChild(greekCol);

    const englishCol = document.createElement("div");
    englishCol.className = "ref-column";
    englishCol.innerHTML = '<div class="ref-column-title">英文字母</div>';
    const englishItems = document.createElement("div");
    englishItems.className = "ref-column-items";
    englishLetters.forEach(l => englishItems.appendChild(createLetterItem(l)));
    englishCol.appendChild(englishItems);
    referenceGrid.appendChild(englishCol);
}

function initBackgroundSymbols() {
    const bgContainer = document.querySelector(".bg-particles");
    if (!bgContainer) return;
    const symbols = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"];
    for (let i = 0; i < 15; i++) {
        const el = document.createElement("div");
        el.className = "floating-symbol";
        el.textContent = symbols[Math.floor(Math.random() * symbols.length)];
        el.style.left = Math.random() * 100 + "vw";
        el.style.animationDuration = 15 + Math.random() * 20 + "s";
        el.style.animationDelay = Math.random() * -30 + "s";
        el.style.fontSize = 1.5 + Math.random() * 2 + "rem";
        bgContainer.appendChild(el);
    }
}

function init() {
    if (isLightMode) document.body.classList.add("light-mode");
    updateThemeToggleLabel();
    initCanvas();
    initBackgroundSymbols();
    setStatus("", "載入模型中...");
    if (referenceGrid) {
        referenceGrid.innerHTML = '<div style="color:var(--text-muted); grid-column:1/-1; text-align:center;">載入中...</div>';
    }
    loadBrowserModel();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    clearCanvas();
});
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", endDraw, { passive: false });
btnClear?.addEventListener("click", clearCanvas);
btnUndo?.addEventListener("click", undoLastStroke);
document.addEventListener("keydown", handleGlobalKeydown);

brushSizeInput?.addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushSizeLabel.textContent = `${brushSize}px`;
});

themeToggle?.addEventListener("click", () => {
    isLightMode = !isLightMode;
    if (isLightMode) {
        document.body.classList.add("light-mode");
        localStorage.setItem("theme", "light");
    } else {
        document.body.classList.remove("light-mode");
        localStorage.setItem("theme", "dark");
    }
    updateThemeToggleLabel();
    clearCanvas();
});

window.addEventListener("DOMContentLoaded", init);
