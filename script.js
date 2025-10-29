// ESM imports (no globals)
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
// Use ESM builds from a CDN that serves correct MIME types:
import * as tf from "https://esm.sh/@tensorflow/tfjs@4.22.0";
import * as poseDetection from "https://esm.sh/@tensorflow-models/pose-detection@3.1.0";

// ----- CONFIG -----
const STATE = { running: false, detector: null, mixers: [], cameraMode: "user" };
const ASSETS = {
  shoulder: "./assets/model1.glb",
  back: "./assets/model2.glb",
  head: "./assets/model3.glb",
};
const CONF = {
  poseEveryNFrames: 2,
  minShoulderScore: 0.4,
  minFaceScore: 0.4,
  zVideo: -10,
  zModels: -5,
  headScaleClamp: [0.6, 2.2],
  shoulderOffset: 0.38,
  backYOffset: 0.2,
  smooth: 0.6,
};

// ----- STATE -----
let video, overlay, octx, renderer, scene, camera, videoPlane, group;
let frame = 0;
let lastPose = null;

function setupThree() {
  const threeEl = document.getElementById("three");
  const rect = threeEl.getBoundingClientRect();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(rect.width, rect.height);
  threeEl.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, rect.width / rect.height, 0.01, 100);
  camera.position.set(0, 0, 0);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.0));

  const tex = new THREE.VideoTexture(video);
  tex.flipY = false;
  if (STATE.cameraMode === "user") { tex.wrapS = THREE.RepeatWrapping; tex.repeat.x = -1; tex.offset.x = 1; }

  const geo = new THREE.PlaneGeometry(1, 1); geo.scale(1, -1, 1);
  videoPlane = new THREE.Mesh(new THREE.MeshBasicMaterial({ map: tex, depthTest: false }));
  videoPlane.geometry = geo;
  scene.add(videoPlane);
  updateVideoPlaneSize();

  group = new THREE.Group(); scene.add(group);

  window.addEventListener("resize", () => {
    const r = threeEl.getBoundingClientRect();
    renderer.setSize(r.width, r.height);
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
    updateVideoPlaneSize();
  });
}

function updateVideoPlaneSize() {
  const threeEl = document.getElementById("three");
  const rect = threeEl.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const fovRad = THREE.MathUtils.degToRad(camera.fov);
  const planeH = Math.abs(2 * Math.tan(fovRad / 2) * CONF.zVideo);
  const planeW = planeH * aspect;
  videoPlane.position.z = CONF.zVideo;
  videoPlane.scale.set(planeW, planeH, 1);
}

function loadGLB(url) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      url,
      (gltf) => {
        const root = gltf.scene || new THREE.Group();
        root.traverse((o) => (o.frustumCulled = false));
        let mixer = null;
        if (gltf.animations?.length) {
          mixer = new THREE.AnimationMixer(root);
          gltf.animations.forEach((clip) => mixer.clipAction(clip).play());
          STATE.mixers.push(mixer);
        }
        resolve({ root, mixer });
      },
      undefined,
      reject
    );
  });
}

const anchors = { shoulderL: null, shoulderR: null, back: null, head: null };

async function setupAnchors() {
  const [m1, m2, m3] = await Promise.all([
    loadGLB(ASSETS.shoulder), loadGLB(ASSETS.back), loadGLB(ASSETS.head)
  ]);
  anchors.shoulderL = m1.root.clone(true);
  anchors.shoulderR = m1.root.clone(true);
  anchors.back = m2.root;
  anchors.head = m3.root;
  [anchors.shoulderL, anchors.shoulderR, anchors.back, anchors.head].forEach(n => {
    n.visible = false; group.add(n);
  });
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: STATE.cameraMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
}

async function setupPose() {
  await tf.ready();
  STATE.detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );
}

const normX = (x, w) => (x / w) * 2 - 1;
const normY = (y, h) => -(y / h) * 2 + 1;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const get = (kps, n) => kps.find(k => k.name === n);
const hasFace = (kps) => {
  const pts = ["nose", "left_eye", "right_eye"].map(n => get(kps, n)).filter(Boolean);
  return pts.some(p => p.score >= CONF.minFaceScore);
};
const smoothSet = (obj, x, y, z) => {
  obj.position.x += (x - obj.position.x) * CONF.smooth;
  obj.position.y += (y - obj.position.y) * CONF.smooth;
  obj.position.z += (z - obj.position.z) * CONF.smooth;
};

async function loop() {
  if (!STATE.running) return;
  requestAnimationFrame(loop);
  frame++;

  if (STATE.detector && video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && frame % CONF.poseEveryNFrames === 0) {
    try {
      lastPose = (await STATE.detector.estimatePoses(video, { flipHorizontal: false }))[0] || null;
    } catch {}
  }

  if (lastPose?.keypoints) {
    const kps = lastPose.keypoints;
    const lS = get(kps, "left_shoulder");
    const rS = get(kps, "right_shoulder");
    const lE = get(kps, "left_ear");
    const rE = get(kps, "right_ear");
    const nose = get(kps, "nose");

    const haveShoulders = lS && rS && lS.score >= CONF.minShoulderScore && rS.score >= CONF.minShoulderScore;
    const faceDetected = hasFace(kps);

    const vw = video.videoWidth, vh = video.videoHeight;

    if (haveShoulders) {
      const cx = (lS.x + rS.x) / 2, cy = (lS.y + rS.y) / 2;
      let x = normX(cx, vw), y = normY(cy, vh);
      if (STATE.cameraMode === "user") x = -x;
      group.position.set(x, y, CONF.zModels);

      const span = Math.abs(normX(lS.x, vw) - normX(rS.x, vw));
      const offset = Math.max(CONF.shoulderOffset, span * 0.35);

      if (anchors.shoulderL && anchors.shoulderR) {
        const show = faceDetected;
        anchors.shoulderL.visible = show; anchors.shoulderR.visible = show;
        if (show) {
          smoothSet(anchors.shoulderL,  offset, 0, 0);
          smoothSet(anchors.shoulderR, -offset, 0, 0);
          anchors.shoulderL.rotation.set(-0.2, Math.PI,  Math.PI * 0.08);
          anchors.shoulderR.rotation.set(-0.2, Math.PI, -Math.PI * 0.08);
          anchors.shoulderL.scale.setScalar(1.0);
          anchors.shoulderR.scale.setScalar(1.0);
        }
      }

      if (anchors.back) {
        const showBack = !faceDetected;
        anchors.back.visible = showBack;
        if (showBack) {
          smoothSet(anchors.back, 0, CONF.backYOffset, -0.2);
          anchors.back.rotation.set(0, Math.PI, 0);
          anchors.back.scale.setScalar(1.2);
        }
      }

      if (anchors.head) {
        const showHead = faceDetected;
        anchors.head.visible = showHead;
        if (showHead) {
          const hx = nose ? nose.x : (lE && rE ? (lE.x + rE.x) / 2 : cx);
          const hy = nose ? nose.y : (lE && rE ? (lE.y + rE.y) / 2 : cy - 40);
          let nx = normX(hx, vw), ny = normY(hy, vh);
          if (STATE.cameraMode === "user") nx = -nx;
          const dx = lE && rE ? Math.abs(normX(lE.x, vw) - normX(rE.x, vw)) : span * 0.6;
          const headScale = clamp(dx * 2.2, CONF.headScaleClamp[0], CONF.headScaleClamp[1]);
          smoothSet(anchors.head, nx - group.position.x, ny - group.position.y + 0.1, -0.1);
          anchors.head.rotation.set(0, Math.PI, 0);
          anchors.head.scale.setScalar(headScale);
        }
      }
    } else {
      if (anchors.back) anchors.back.visible = false;
      if (anchors.head) anchors.head.visible = false;
      if (anchors.shoulderL) anchors.shoulderL.visible = false;
      if (anchors.shoulderR) anchors.shoulderR.visible = false;
    }
  }

  STATE.mixers.forEach((m) => m.update(1 / 60));
  if (videoPlane?.material.map) videoPlane.material.map.needsUpdate = true;
  renderer.render(scene, camera);
}

async function start() {
  document.getElementById("ui").style.display = "none";
  video = document.getElementById("video");
  overlay = document.getElementById("overlay");
  octx = overlay.getContext("2d");

  await setupCamera();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;

  setupThree();
  await tf.ready();
  await setupPose();
  await setupAnchors();

  STATE.running = true;
  loop();
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("start").addEventListener("click", start, { passive: true });
});
