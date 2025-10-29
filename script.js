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
  inited: false,           // prevents duplicate init
  cameraMode: "environment", // rear by default
  landmarker: null,
  mixers: [],
  stream: null
};

/* ---------------- Globals ---------------- */
let video, overlay, octx, renderer, scene, camera, videoPlane, group;
let frame = 0, lastResult = null;
const IDX = { NOSE: 0, LEFT_EYE: 2, RIGHT_EYE: 5, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12 };
const anchors = { shoulderL: null, shoulderR: null, back: null, head: null };

/* ---------------- Shortcuts ---------------- */
const $ = (sel) => document.querySelector(sel);
const normX = (x) => x * 2 - 1;
const normY = (y) => -y * 2 + 1;
const clamp = (v,a,b)=>Math.min(b,Math.max(a,v));
const smoothSet = (o,x,y,z)=>{ o.position.x += (x-o.position.x)*CONF.smooth; o.position.y += (y-o.position.y)*CONF.smooth; o.position.z += (z-o.position.z)*CONF.smooth; };
const facePresent = (lm)=> !!(lm[IDX.NOSE] || lm[IDX.LEFT_EYE] || lm[IDX.RIGHT_EYE]);
const setText = (id, val) => { const el=$(id); if (el) el.textContent = val; };
function toast(msg, ms=2200){ const t=$("#toast"); t.textContent=msg; t.hidden=false; setTimeout(()=>t.hidden=true, ms); }

/* ---------------- Diagnostics ---------------- */
function updateDiagnostics({httpsOk, apiOk, perm, status, err}) {
  setText("#d-https", httpsOk ? "OK" : "Not secure (HTTPS required)");
  setText("#d-api", apiOk ? "Available" : "Unavailable");
  setText("#d-perm", perm ?? "Unknown");
  setText("#d-status", status ?? "—");
  setText("#d-err", err ?? "None");
  $("#banner").hidden = !(status === "Camera blocked" || !httpsOk || !apiOk);
}

/* ---------------- Three ---------------- */
function setupThree() {
  if (renderer) return; // idempotent

  const threeEl = $("#three");
  const r = threeEl.getBoundingClientRect();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(r.width, r.height);
  threeEl.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, r.width / r.height, 0.01, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.0));

  const tex = new THREE.VideoTexture(video);
  tex.flipY = false;
  if (STATE.cameraMode === "user") { tex.wrapS = THREE.RepeatWrapping; tex.repeat.x = -1; tex.offset.x = 1; }
  const geo = new THREE.PlaneGeometry(1, 1); geo.scale(1, -1, 1);
  const mat = new THREE.MeshBasicMaterial({ map: tex, depthTest: false });
  videoPlane = new THREE.Mesh(geo, mat);
  scene.add(videoPlane); updateVideoPlaneSize();

  group = new THREE.Group(); scene.add(group);

  addEventListener("resize", () => {
    const rr = threeEl.getBoundingClientRect();
    renderer.setSize(rr.width, rr.height);
    camera.aspect = rr.width / rr.height;
    camera.updateProjectionMatrix();
    updateVideoPlaneSize();
  });
}
function updateVideoPlaneSize() {
  if (!camera || !videoPlane) return;
  const threeEl = $("#three");
  const rect = threeEl.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const h = Math.abs(2 * Math.tan(fov / 2) * CONF.zVideo);
  videoPlane.position.z = CONF.zVideo;
  videoPlane.scale.set(h * aspect, h, 1);
}

/* ---------------- GLB loader with fallback ---------------- */
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

/* ---------------- Anchors (idempotent) ---------------- */
let anchorsReady = false;
async function setupAnchors() {
  if (anchorsReady) return;
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
  anchorsReady = true;
}

/* ---------------- Camera (explicit permission flow) ---------------- */
async function queryPermission() {
  try {
    // Not supported in iOS Safari; handled gracefully.
    const res = await navigator.permissions.query({ name: "camera" });
    return res.state; // 'granted' | 'denied' | 'prompt'
  } catch {
    return "unknown";
  }
}

async function grantCamera() {
  const httpsOk = isSecureContext;
  const apiOk = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  if (!httpsOk || !apiOk) {
    updateDiagnostics({ httpsOk, apiOk, perm: "unknown", status: "Camera blocked", err: !httpsOk ? "Not HTTPS" : "API unavailable" });
    throw new Error(!httpsOk ? "This page is not served over HTTPS." : "Camera API not available in this browser.");
  }

  const constraints = { video: { facingMode: STATE.cameraMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
  try {
    STATE.stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = STATE.stream;
    await video.play(); // user gesture already happened (Grant button)
    $("#start").disabled = false;
    updateDiagnostics({ httpsOk, apiOk, perm: "granted or active", status: "Camera ready" });
    toast("Camera granted");
    return true;
  } catch (e) {
    let msg = "Camera permission denied or unavailable.";
    if (e.name === "NotAllowedError") msg = "Permission denied. Enable camera for this site in Settings.";
    if (e.name === "NotFoundError") msg = "No camera found.";
    if (e.name === "NotReadableError") msg = "Camera is in use by another app.";
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

  if (STATE.landmarker && video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && frame % CONF.poseEveryNFrames === 0) {
    lastResult = STATE.landmarker.detectForVideo(video, performance.now());
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

/* ---------------- Start (idempotent, no duplicates) ---------------- */
async function start() {
  if (STATE.inited) { toast("Already running"); return; }
  STATE.inited = true;

  $("#error").textContent = "";
  $("#ui").style.display = "none";

  video = $("#video");
  overlay = $("#overlay");
  octx = overlay.getContext("2d");

  overlay.width = video.videoWidth || 1280;
  overlay.height = video.videoHeight || 720;

  setupThree();
  await setupPose();
  await setupAnchors();

  STATE.running = true;
  loop();
}

/* ---------------- UI and Permission Flow ---------------- */
window.addEventListener("DOMContentLoaded", async () => {
  // Diagnostics
  const httpsOk = isSecureContext;
  const apiOk = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  let perm = await queryPermission().catch(()=> "unknown");
  updateDiagnostics({ httpsOk, apiOk, perm, status: "Idle" });

  // Camera toggle
  const rearBox = $("#rear");
  rearBox.checked = (STATE.cameraMode === "environment");
  rearBox.addEventListener("change", (e) => {
    STATE.cameraMode = e.target.checked ? "environment" : "user";
    toast(`Camera: ${STATE.cameraMode === "environment" ? "rear" : "front"}`);
  }, { passive:true });

  // Grant camera first (explicit user gesture)
  $("#grant").addEventListener("click", async () => {
    try {
      setText("#d-status", "Requesting camera…");
      await grantCamera();
      setText("#d-status", "Camera ready");
      $("#start").disabled = false;
    } catch (e) {
      $("#error").textContent = e.message;
      setText("#d-status", "Camera blocked");
    }
  }, { passive:true });

  // Then start AR (second gesture; ensures iOS autoplay compliance)
  $("#start").addEventListener("click", start, { passive:true });
});
