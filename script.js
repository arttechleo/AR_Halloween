// Three.js + GLTFLoader via import map
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// MediaPipe Tasks Pose (ESM)
import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/vision_bundle.mjs";

/* ---- Assets (consistent names) ---- */
const ASSETS = {
  model1: "./assets/model1.glb", // shoulders when face detected
  model2: "./assets/model2.glb", // back when no face
  model3: "./assets/model3.glb", // covers head when face detected
};

/* ---- Config ---- */
const CONF = {
  zVideo: -10, zModels: -5,
  smooth: 0.6,
  poseEveryNFrames: 2,
  shoulderOffsetMin: 0.38,
  backYOffset: 0.2,
  headScaleClamp: [0.6, 2.2]
};

/* ---- State ---- */
const STATE = {
  running: false,
  cameraMode: "environment", // rear by default
  landmarker: null,
  mixers: []
};

/* ---- Globals ---- */
let video, overlay, octx, renderer, scene, camera, videoPlane, group;
let frame = 0, lastResult = null;

const IDX = { NOSE: 0, LEFT_EYE: 2, RIGHT_EYE: 5, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12 };
const anchors = { shoulderL: null, shoulderR: null, back: null, head: null };

/* ---- Utils ---- */
const $ = (s) => document.querySelector(s);
const normX = (x) => x * 2 - 1;
const normY = (y) => -y * 2 + 1;
const clamp = (v,a,b)=>Math.min(b,Math.max(a,v));
const smoothSet = (o,x,y,z)=>{ o.position.x += (x-o.position.x)*CONF.smooth; o.position.y += (y-o.position.y)*CONF.smooth; o.position.z += (z-o.position.z)*CONF.smooth; };
const facePresent = (lm)=> !!(lm[IDX.NOSE] || lm[IDX.LEFT_EYE] || lm[IDX.RIGHT_EYE]);
function toast(msg, ms=2000){ const t=$("#toast"); t.textContent=msg; t.hidden=false; setTimeout(()=>t.hidden=true, ms); }

/* ---- Three ---- */
function setupThree() {
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
  const threeEl = $("#three");
  const rect = threeEl.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const h = Math.abs(2 * Math.tan(fov / 2) * CONF.zVideo);
  videoPlane.position.z = CONF.zVideo;
  videoPlane.scale.set(h * aspect, h, 1);
}

/* ---- GLB loader with fallback ---- */
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

/* ---- Anchors ---- */
async function setupAnchors() {
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
}

/* ---- Camera (robust errors) ---- */
async function setupCamera() {
  if (!isSecureContext) throw new Error("Not HTTPS. Camera is blocked.");
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API not available.");

  const constraints = { video: { facingMode: STATE.cameraMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    let msg = "Camera permission denied or unavailable.";
    if (e.name === "NotAllowedError") msg = "Permission denied. Enable camera for this site.";
    if (e.name === "NotFoundError") msg = "No camera found.";
    if (e.name === "NotReadableError") msg = "Camera is in use by another app.";
    throw new Error(msg);
  }
}

/* ---- MediaPipe Pose ---- */
async function setupPose() {
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

/* ---- Main loop ---- */
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

/* ---- Start ---- */
async function start() {
  $("#error").textContent = "";
  $("#ui").style.display = "none";

  video = $("#video");
  overlay = $("#overlay");
  octx = overlay.getContext("2d");

  try {
    await setupCamera();
  } catch (e) {
    $("#ui").style.display = "flex";
    $("#error").textContent = e.message;
    return;
  }

  overlay.width = video.videoWidth; overlay.height = video.videoHeight;

  setupThree();
  await setupPose();
  await setupAnchors();

  STATE.running = true;
  loop();
}

/* ---- UI ---- */
window.addEventListener("DOMContentLoaded", () => {
  const rearBox = $("#rear");
  rearBox.checked = (STATE.cameraMode === "environment");
  rearBox.addEventListener("change", (e) => {
    STATE.cameraMode = e.target.checked ? "environment" : "user";
    toast(`Camera: ${STATE.cameraMode === "environment" ? "rear" : "front"}`);
  }, { passive:true });

  $("#start").addEventListener("click", start, { passive:true });
});
