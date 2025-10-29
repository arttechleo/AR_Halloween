// Three.js + GLTFLoader (via import map)
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
// MediaPipe Tasks Pose (ESM)
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/vision_bundle.mjs";

/* ---------------- Config ---------------- */
const ASSETS = {
  model1: "./assets/model1.glb", // shoulders when face detected
  model2: "./assets/model2.glb", // back when no face
  model3: "./assets/model3.glb", // covers head when face detected
};
const CONF = {
  zVideo: -10, zModels: -5,
  smooth: 0.6,
  poseEveryNFrames: 2,
  shoulderOffsetMin: 0.38,
  backYOffset: 0.2,
  headScaleClamp: [0.6, 2.2]
};

/* ---------------- State ---------------- */
const STATE = {
  running: false,
  inited: false,
  cameraMode: "environment",   // rear by default
  landmarker: null,
  mixers: [],
  stream: null,
  anchorsReady: false
};

/* ---------------- DOM refs (filled by ensureDom) ---------------- */
const DOM = {
  video: null,
  overlay: null,
  three: null,
  ui: null,
  startBtn: null,
  grantBtn: null,
  rearBox: null,
  openSafariBtn: null,
};
const $ = (s) => document.querySelector(s);
function ensureDom() {
  if (!DOM.video)       DOM.video = $("#video");
  if (!DOM.overlay)     DOM.overlay = $("#overlay");
  if (!DOM.three)       DOM.three = $("#three");
  if (!DOM.ui)          DOM.ui = $("#ui");
  if (!DOM.startBtn)    DOM.startBtn = $("#start");
  if (!DOM.grantBtn)    DOM.grantBtn = $("#grant");
  if (!DOM.rearBox)     DOM.rearBox = $("#rear");
  if (!DOM.openSafariBtn) DOM.openSafariBtn = $("#openSafari");
  if (!DOM.video) throw new Error("Video element not found in DOM.");
}

/* ---------------- Utilities ---------------- */
const normX = (x) => x * 2 - 1;
const normY = (y) => -y * 2 + 1;
const clamp = (v,a,b)=>Math.min(b,Math.max(a,v));
const smoothSet = (o,x,y,z)=>{ o.position.x += (x-o.position.x)*CONF.smooth; o.position.y += (y-o.position.y)*CONF.smooth; o.position.z += (z-o.position.z)*CONF.smooth; };
const setText = (id, val) => { const el=$(id); if (el) el.textContent = val; };
function toast(msg, ms=2200){ const t=$("#toast"); if(!t) return; t.textContent=msg; t.hidden=false; setTimeout(()=>t.hidden=true, ms); }
const IDX = { NOSE: 0, LEFT_EYE: 2, RIGHT_EYE: 5, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12 };
const facePresent = (lm)=> !!(lm[IDX.NOSE] || lm[IDX.LEFT_EYE] || lm[IDX.RIGHT_EYE]);
function inAppBrowser(){ const ua=navigator.userAgent||""; return /(FBAN|FBAV|Instagram|Twitter|Line|WeChat|OKApp|TikTok|Snapchat|Pinterest)/i.test(ua); }

/* ---------------- Diagnostics ---------------- */
function updateDiagnostics({httpsOk, apiOk, perm, status, err}) {
  setText("#d-https", httpsOk ? "OK" : "Not secure (HTTPS required)");
  setText("#d-api", apiOk ? "Available" : "Unavailable");
  setText("#d-perm", perm ?? "Unknown");
  setText("#d-status", status ?? "—");
  setText("#d-err", err ?? "None");
  const banner = $("#banner");
  if (banner) banner.hidden = !(status === "Camera blocked" || !httpsOk || !apiOk);
  const openSafari = $("#openSafari");
  if (openSafari) openSafari.hidden = !inAppBrowser();
}

/* ---------------- Three ---------------- */
let renderer, scene, camera, videoPlane, group, frame=0, lastResult=null;
function setupThree() {
  if (renderer) return;
  ensureDom();
  const r = DOM.three.getBoundingClientRect();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(r.width, r.height);
  DOM.three.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, r.width / r.height, 0.01, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.0));

  const tex = new THREE.VideoTexture(DOM.video);
  tex.flipY = false;
  if (STATE.cameraMode === "user") { tex.wrapS = THREE.RepeatWrapping; tex.repeat.x = -1; tex.offset.x = 1; }
  const geo = new THREE.PlaneGeometry(1, 1); geo.scale(1, -1, 1);
  const mat = new THREE.MeshBasicMaterial({ map: tex, depthTest: false });
  videoPlane = new THREE.Mesh(geo, mat);
  scene.add(videoPlane); updateVideoPlaneSize();

  group = new THREE.Group(); scene.add(group);

  addEventListener("resize", () => {
    const rr = DOM.three.getBoundingClientRect();
    renderer.setSize(rr.width, rr.height);
    camera.aspect = rr.width / rr.height;
    camera.updateProjectionMatrix();
    updateVideoPlaneSize();
  });
}
function updateVideoPlaneSize() {
  if (!camera || !videoPlane) return;
  const rect = DOM.three.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const h = Math.abs(2 * Math.tan(fov / 2) * CONF.zVideo);
  videoPlane.position.z = CONF.zVideo;
  videoPlane.scale.set(h * aspect, h, 1);
}

/* ---------------- GLB loader (with fallback cube) ---------------- */
function loadGLB(url, label) {
  return new Promise((resolve) => {
    new GLTFLoader().load(
      url,
      (g) => {
        const root = g.scene || new THREE.Group();
        root.traverse(o=>o.frustumCulled=false);
        if (g.animations?.length) {
          const mx = new THREE.AnimationMixer(root);
          g.animations.forEach(c=>mx.clipAction(c).play());
          STATE.mixers.push(mx);
        }
        resolve(root);
      },
      undefined,
      () => {
        const geo = new THREE.BoxGeometry(0.3,0.3,0.3);
        const mat = new THREE.MeshNormalMaterial();
        const box = new THREE.Mesh(geo,mat);
        box.userData.fallback = label;
        resolve(box);
      }
    );
  });
}

/* ---------------- Anchors (load once) ---------------- */
const anchors = { shoulderL: null, shoulderR: null, back: null, head: null };
async function setupAnchors() {
  if (STATE.anchorsReady) return;
  const [m1, m2, m3] = await Promise.all([
    loadGLB(ASSETS.model1, "model1"),
    loadGLB(ASSETS.model2, "model2"),
    loadGLB(ASSETS.model3, "model3"),
  ]);
  anchors.shoulderL = m1.clone(true);
  anchors.shoulderR = m1.clone(true);
  anchors.back = m2;
  anchors.head = m3;
  [anchors.shoulderL, anchors.shoulderR, anchors.back, anchors.head].forEach(n => { n.visible = false; group.add(n); });
  STATE.anchorsReady = true;
}

/* ---------------- Permission helpers ---------------- */
async function queryPermission() {
  try { const res = await navigator.permissions.query({ name: "camera" }); return res.state; }
  catch { return "unknown"; } // iOS Safari often throws
}

/* --- robust camera acquisition: rear -> rear-preferred -> any camera --- */
async function getCameraStream() {
  const tryConstraint = async (c) => {
    try { return await navigator.mediaDevices.getUserMedia(c); }
    catch (e) {
      if (e.name === "OverconstrainedError" || e.name === "NotFoundError") return null;
      throw e; // NotAllowed/NotReadable etc.
    }
  };
  // 1) exact rear
  let s = await tryConstraint({ video: { facingMode: { exact: "environment" } }, audio: false });
  if (s) return s;
  // 2) prefer rear
  s = await tryConstraint({ video: { facingMode: "environment" }, audio: false });
  if (s) return s;
  // 3) any
  s = await tryConstraint({ video: true, audio: false });
  return s;
}

/* Request camera with full UX + fallbacks */
async function grantCamera() {
  ensureDom(); // <<< FIX: make sure DOM.video exists before using it
  const httpsOk = isSecureContext;
  const apiOk = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  const perm = await queryPermission();

  if (inAppBrowser()) {
    updateDiagnostics({ httpsOk, apiOk, perm, status: "Camera blocked (in-app browser)", err: "Open in Safari" });
    throw new Error("Open this page in Safari (in-app browsers often block camera).");
  }
  if (!httpsOk || !apiOk) {
    updateDiagnostics({ httpsOk, apiOk, perm, status: "Camera blocked", err: !httpsOk ? "Not HTTPS" : "API unavailable" });
    throw new Error(!httpsOk ? "This page is not served over HTTPS." : "Camera API not available.");
  }

  try {
    const stream = await getCameraStream();
    if (!stream) {
      updateDiagnostics({ httpsOk, apiOk, perm, status: "Camera blocked", err: "No usable camera found" });
      throw new Error("No usable camera found.");
    }
    STATE.stream = stream;
    // --- SAFETY: video element must exist
    if (!DOM.video) throw new Error("Video element missing from DOM.");
    DOM.video.srcObject = stream;
    await DOM.video.play(); // user gesture already happened (Grant button)
    if (DOM.startBtn) DOM.startBtn.disabled = false;
    updateDiagnostics({ httpsOk, apiOk, perm: "granted or active", status: "Camera ready" });
    toast("Camera granted");
    return true;
  } catch (e) {
    const msg = (e && e.message) || "Camera permission denied or unavailable.";
    updateDiagnostics({ httpsOk, apiOk, perm: "denied", status: "Camera blocked", err: msg });
    throw new Error(msg);
  }
}

/* ---------------- MediaPipe Pose ---------------- */
async function setupPose() {
  if (STATE.landmarker) return;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/wasm"
  );
  STATE.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });
}

/* ---------------- Main loop ---------------- */
async function loop() {
  if (!STATE.running) return;
  requestAnimationFrame(loop);
  frame++;

  if (STATE.landmarker && DOM.video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && frame % CONF.poseEveryNFrames === 0) {
    lastResult = STATE.landmarker.detectForVideo(DOM.video, performance.now());
  }

  const poses = lastResult?.landmarks;
  if (poses && poses.length) {
    const lm = poses[0];
    const lS = lm[IDX.LEFT_SHOULDER], rS = lm[IDX.RIGHT_SHOULDER];
    if (lS && rS) {
      const cx = (lS.x + rS.x) / 2, cy = (lS.y + rS.y) / 2;
      let gx = normX(cx), gy = normY(cy); if (STATE.cameraMode === "user") gx = -gx;
      group.position.set(gx, gy, CONF.zModels);

      const span = Math.abs(normX(lS.x) - normX(rS.x));
      const offset = Math.max(CONF.shoulderOffsetMin, span * 0.35);
      const hasFace = facePresent(lm);

      // model1.glb on shoulders when face detected
      [anchors.shoulderL, anchors.shoulderR].forEach(n => n.visible = !!hasFace);
      if (hasFace) {
        smoothSet(anchors.shoulderL,  offset, 0, 0);
        smoothSet(anchors.shoulderR, -offset, 0, 0);
        anchors.shoulderL.rotation.set(-0.2, Math.PI,  Math.PI * 0.08);
        anchors.shoulderR.rotation.set(-0.2, Math.PI, -Math.PI * 0.08);
        anchors.shoulderL.scale.setScalar(1.0);
        anchors.shoulderR.scale.setScalar(1.0);
      }

      // model2.glb on back when no face
      anchors.back.visible = !hasFace;
      if (!hasFace) {
        smoothSet(anchors.back, 0, CONF.backYOffset, -0.2);
        anchors.back.rotation.set(0, Math.PI, 0);
        anchors.back.scale.setScalar(1.2);
      }

      // model3.glb covering head when face detected
      anchors.head.visible = hasFace;
      if (hasFace) {
        const nose = lm[IDX.NOSE] || { x: cx, y: cy - 0.04 };
        let nx = normX(nose.x), ny = normY(nose.y);
        if (STATE.cameraMode === "user") nx = -nx;
        const le = lm[IDX.LEFT_EYE], re = lm[IDX.RIGHT_EYE];
        const dx = (le && re) ? Math.abs(normX(le.x) - normX(re.x)) : span * 0.6;
        const headScale = clamp(dx * 2.2, CONF.headScaleClamp[0], CONF.headScaleClamp[1]);
        smoothSet(anchors.head, nx - group.position.x, ny - group.position.y + 0.1, -0.1);
        anchors.head.rotation.set(0, Math.PI, 0);
        anchors.head.scale.setScalar(headScale);
      }
    } else {
      Object.values(anchors).forEach(n => n && (n.visible = false));
    }
  }

  STATE.mixers.forEach(m => m.update(1/60));
  if (videoPlane?.material.map) videoPlane.material.map.needsUpdate = true;
  renderer.render(scene, camera);
}

/* ---------------- Start (guarded) ---------------- */
async function start() {
  if (STATE.inited) { toast("Already running"); return; }
  STATE.inited = true;

  ensureDom();
  $("#error").textContent = "";
  DOM.ui.style.display = "none";

  // set overlay dimensions early (safe defaults)
  DOM.overlay.width = 1280;
  DOM.overlay.height = 720;

  setupThree();
  await setupPose();
  await setupAnchors();

  STATE.running = true;
  loop();
}

/* ---------------- UI ---------------- */
window.addEventListener("DOMContentLoaded", async () => {
  ensureDom(); // <<< FIX: cache DOM before any button actions
  const httpsOk = isSecureContext;
  const apiOk = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  let perm = await queryPermission().catch(()=> "unknown");
  updateDiagnostics({ httpsOk, apiOk, perm, status: "Idle" });

  DOM.rearBox.checked = true;
  DOM.rearBox.addEventListener("change", (e) => {
    STATE.cameraMode = e.target.checked ? "environment" : "user";
    toast(`Camera: ${STATE.cameraMode === "environment" ? "rear" : "front"}`);
  }, { passive:true });

  if (DOM.openSafariBtn) {
    DOM.openSafariBtn.addEventListener("click", () => {
      // best-effort prompt to open in Safari (still same URL, but instructs user)
      alert("If camera stays blocked, copy this URL and open in Safari directly.");
    });
  }

  DOM.grantBtn.addEventListener("click", async () => {
    try {
      setText("#d-status", "Requesting camera…");
      await grantCamera();
      setText("#d-status", "Camera ready");
      DOM.startBtn.disabled = false;
    } catch (e) {
      $("#error").textContent = e.message;
      setText("#d-status", "Camera blocked");
    }
  }, { passive:true });

  DOM.startBtn.addEventListener("click", start, { passive:true });
});
