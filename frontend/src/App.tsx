import { FormEvent, PointerEvent, useCallback, useEffect, useState } from "react";

type Detection = {
  class_name: string;
  predicted_class: string;
  temporal_candidate: string;
  final_classification: string | null;
  confidence: number;
  classification_confidence: number | null;
  temporal_support: number;
  verification_status: "verified" | "primary_only" | "uncertain";
  verification_score: number | null;
  verification_alternative: string | null;
  observations: string[];
  object_id: string | null;
  track_id: number | null;
  first_seen_frame: number | null;
  last_seen_frame: number | null;
  track_observation_count: number;
  prediction_history: Array<{
    frame_index: number;
    predicted_class: string;
    confidence: number;
    verification_status: string;
  }>;
  movement_pixels: number | null;
  recognition_status:
    | "VERIFIED_RESIDENT"
    | "VERIFIED_VISITOR"
    | "VERIFIED_STAFF"
    | "UNKNOWN"
    | "UNVERIFIED"
    | "FACE_NOT_VISIBLE"
    | "IDENTIFYING";
  recognized_person_id: number | null;
  recognized_name: string | null;
  recognized_category: "resident" | "visitor" | "authorized_staff" | null;
  recognition_similarity: number | null;
  box: number[];
};

type Camera = {
  id: string;
  name: string;
  status: "connecting" | "online" | "offline" | "error" | "stopped";
  source_type: "webcam" | "network";
  detections: Detection[];
  timestamp: string | null;
  error: string | null;
};

type DetectionEvent = {
  type: "detections";
  camera_id: string;
  camera_name: string;
  timestamp: string;
  detections: Detection[];
};

type ZonePoint = { x: number; y: number };

type Zone = {
  id: string;
  camera_id: string;
  name: string;
  polygon: ZonePoint[];
  enabled: boolean;
  loiter_seconds: number;
  event_cooldown_seconds: number;
};

type ZoneDefaults = {
  min_track_age_seconds: number;
  entry_frames: number;
  exit_frames: number;
  loiter_seconds: number;
  event_cooldown_seconds: number;
  track_loss_tolerance_seconds: number;
  track_match_distance_fraction: number;
};

type ZoneRuleEvent = {
  type: "zone_rule";
  camera_id: string;
  camera_name: string;
  zone_id: string;
  zone_name: string;
  event_type: "entry" | "exit" | "loitering";
  track_id: number | null;
  object_label: string;
  timestamp: string;
};

type PersonRecord = {
  id: number;
  display_name: string;
  category: "resident" | "visitor" | "authorized_staff";
  role: string;
  active: boolean;
  consented_at: string | null;
  enrollment_count: number;
  created_at: string;
};

type ReviewEvent = {
  id: string;
  camera_id: string;
  camera_name: string;
  timestamp: string;
  detections: Detection[];
  created_at: string;
  reviewed: boolean;
  reviewed_at: string | null;
  person_id: number | null;
  person_name: string | null;
  person_role: string | null;
};

type RecognitionEvent = {
  id: number;
  camera_id: string;
  tracking_id: number;
  timestamp: string;
  status: Detection["recognition_status"];
  similarity: number | null;
  person_id: number | null;
  person_name: string | null;
  person_category: PersonRecord["category"] | null;
};

function App() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [zoneEvents, setZoneEvents] = useState<ZoneRuleEvent[]>([]);
  const [zones, setZones] = useState<Record<string, Zone[]>>({});
  const [zoneDefaults, setZoneDefaults] = useState<ZoneDefaults>({
    min_track_age_seconds: 1,
    entry_frames: 2,
    exit_frames: 2,
    loiter_seconds: 30,
    event_cooldown_seconds: 30,
    track_loss_tolerance_seconds: 2,
    track_match_distance_fraction: 0.03,
  });
  const [drawingCamera, setDrawingCamera] = useState<string | null>(null);
  const [draftPoints, setDraftPoints] = useState<ZonePoint[]>([]);
  const [editingZone, setEditingZone] = useState<Zone | null>(null);
  const [zoneName, setZoneName] = useState("");
  const [zoneLoiterSeconds, setZoneLoiterSeconds] = useState(30);
  const [zoneCooldownSeconds, setZoneCooldownSeconds] = useState(30);
  const [draggedVertex, setDraggedVertex] = useState<number | null>(null);
  const [feedDimensions, setFeedDimensions] = useState<
    Record<string, { width: number; height: number }>
  >({});
  const [people, setPeople] = useState<PersonRecord[]>([]);
  const [reviewEvents, setReviewEvents] = useState<ReviewEvent[]>([]);
  const [recognitionEvents, setRecognitionEvents] = useState<RecognitionEvent[]>([]);
  const [recognitionStatus, setRecognitionStatus] = useState("checking");
  const [reviewSelections, setReviewSelections] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [source, setSource] = useState("0");
  const [personName, setPersonName] = useState("");
  const [personCategory, setPersonCategory] = useState<PersonRecord["category"]>("resident");
  const [enrollmentFiles, setEnrollmentFiles] = useState<Record<number, File[]>>({});
  const [consentChecks, setConsentChecks] = useState<Record<number, boolean>>({});
  const [editingPersonId, setEditingPersonId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [editCategory, setEditCategory] = useState<PersonRecord["category"]>("resident");
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);

  const refreshCameras = useCallback(async () => {
    try {
      const response = await fetch("/api/cameras");
      if (!response.ok) throw new Error(`Camera list failed (${response.status})`);
      setCameras((await response.json()) as Camera[]);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not reach SentinelAI.");
    }
  }, []);

  const refreshPeopleData = useCallback(async () => {
    try {
      const [peopleResponse, eventsResponse, statusResponse, historyResponse] = await Promise.all([
        fetch("/api/people"),
        fetch("/api/review-events"),
        fetch("/api/recognition/status"),
        fetch("/api/recognition/events"),
      ]);
      if (!peopleResponse.ok || !eventsResponse.ok || !statusResponse.ok || !historyResponse.ok) {
        throw new Error("Could not load people or saved review events.");
      }
      setPeople((await peopleResponse.json()) as PersonRecord[]);
      setReviewEvents((await eventsResponse.json()) as ReviewEvent[]);
      const status = (await statusResponse.json()) as { status: string };
      setRecognitionStatus(status.status);
      setRecognitionEvents((await historyResponse.json()) as RecognitionEvent[]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load saved data.");
    }
  }, []);

  useEffect(() => {
    void refreshCameras();
    void refreshPeopleData();
    const timer = window.setInterval(() => void refreshCameras(), 2000);
    return () => window.clearInterval(timer);
  }, [refreshCameras, refreshPeopleData]);

  useEffect(() => {
    let socket: WebSocket | undefined;
    let retryTimer: number;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/events`);
      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        if (!disposed) retryTimer = window.setTimeout(connect, 1500);
      };
      socket.onerror = () => socket?.close();
      socket.onmessage = (message: MessageEvent<string>) => {
        const event = JSON.parse(message.data) as DetectionEvent | ZoneRuleEvent;
        if (event.type === "zone_rule") {
          setZoneEvents((current) => [event, ...current].slice(0, 40));
          return;
        }
        setEvents((current) => [event, ...current].slice(0, 40));
        setCameras((current) =>
          current.map((camera) =>
            camera.id === event.camera_id
              ? { ...camera, detections: event.detections, timestamp: event.timestamp }
              : camera,
          ),
        );
      };
    };

    connect();
    return () => {
      disposed = true;
      window.clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    const loadZones = async () => {
      try {
        const entries = await Promise.all(
          cameras.map(async (camera) => {
            const response = await fetch(`/api/cameras/${camera.id}/zones`);
            if (!response.ok) throw new Error(`Zone list failed (${response.status})`);
            return [camera.id, (await response.json()) as Zone[]] as const;
          }),
        );
        if (!disposed) setZones(Object.fromEntries(entries));
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Could not load camera zones.");
      }
    };
    void loadZones();
    return () => {
      disposed = true;
    };
  }, [cameras.map((camera) => camera.id).join(",")]);

  useEffect(() => {
    let disposed = false;
    void fetch("/api/zones/defaults")
      .then(async (response) => {
        if (!response.ok) throw new Error(`Zone defaults failed (${response.status})`);
        return (await response.json()) as ZoneDefaults;
      })
      .then((defaults) => {
        if (!disposed) setZoneDefaults(defaults);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Could not load zone defaults.");
      });
    return () => {
      disposed = true;
    };
  }, []);

  async function addCamera(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    try {
      const response = await fetch("/api/cameras", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, source }),
      });
      if (!response.ok) {
        const body = (await response.json()) as { detail?: string };
        throw new Error(body.detail ?? `Camera setup failed (${response.status})`);
      }
      setName("");
      await refreshCameras();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add camera.");
    }
  }

  async function removeCamera(camera: Camera) {
    setError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error(`Camera removal failed (${response.status})`);
      setCameras((current) => current.filter((item) => item.id !== camera.id));
      setZones((current) => {
        const next = { ...current };
        delete next[camera.id];
        return next;
      });
      if (drawingCamera === camera.id) cancelZoneEdit();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not remove camera.");
    }
  }

  function cancelZoneEdit() {
    setDrawingCamera(null);
    setDraftPoints([]);
    setEditingZone(null);
    setDraggedVertex(null);
  }

  function beginZoneDrawing(cameraId: string) {
    setDrawingCamera(cameraId);
    setDraftPoints([]);
    setEditingZone(null);
    setZoneName("");
    setZoneLoiterSeconds(zoneDefaults.loiter_seconds);
    setZoneCooldownSeconds(zoneDefaults.event_cooldown_seconds);
  }

  function beginZoneEdit(cameraId: string, zone: Zone) {
    setDrawingCamera(cameraId);
    setEditingZone(zone);
    setDraftPoints(zone.polygon.map((point) => ({ ...point })));
    setZoneName(zone.name);
    setZoneLoiterSeconds(zone.loiter_seconds);
    setZoneCooldownSeconds(zone.event_cooldown_seconds);
  }

  function pointFromPointer(event: PointerEvent<SVGSVGElement>): ZonePoint {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    };
  }

  function addZonePoint(event: PointerEvent<SVGSVGElement>) {
    if (drawingCamera === null || editingZone) return;
    setDraftPoints((current) => [...current, pointFromPointer(event)]);
  }

  function moveZoneVertex(event: PointerEvent<SVGSVGElement>) {
    if (draggedVertex === null) return;
    const point = pointFromPointer(event);
    setDraftPoints((current) =>
      current.map((existing, index) => (index === draggedVertex ? point : existing)),
    );
  }

  async function saveZone(cameraId: string) {
    setError("");
    if (draftPoints.length < 3) {
      setError("A zone needs at least three polygon points.");
      return;
    }
    try {
      const payload = {
        name: zoneName,
        polygon: draftPoints,
        enabled: editingZone?.enabled ?? true,
        loiter_seconds: zoneLoiterSeconds,
        event_cooldown_seconds: zoneCooldownSeconds,
      };
      const url = editingZone
        ? `/api/cameras/${cameraId}/zones/${editingZone.id}`
        : `/api/cameras/${cameraId}/zones`;
      const response = await fetch(url, {
        method: editingZone ? "PUT" : "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const body = (await response.json()) as { detail?: string };
        throw new Error(body.detail ?? `Zone save failed (${response.status})`);
      }
      const saved = (await response.json()) as Zone;
      setZones((current) => ({
        ...current,
        [cameraId]: [
          ...(current[cameraId] ?? []).filter((zone) => zone.id !== saved.id),
          saved,
        ],
      }));
      cancelZoneEdit();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save zone.");
    }
  }

  async function toggleZone(cameraId: string, zone: Zone) {
    setError("");
    try {
      const response = await fetch(`/api/cameras/${cameraId}/zones/${zone.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ...zone, enabled: !zone.enabled }),
      });
      if (!response.ok) throw new Error(`Zone update failed (${response.status})`);
      const updated = (await response.json()) as Zone;
      setZones((current) => ({
        ...current,
        [cameraId]: (current[cameraId] ?? []).map((item) =>
          item.id === zone.id ? updated : item,
        ),
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not update zone.");
    }
  }

  async function deleteZone(cameraId: string, zone: Zone) {
    setError("");
    try {
      const response = await fetch(`/api/cameras/${cameraId}/zones/${zone.id}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error(`Zone deletion failed (${response.status})`);
      setZones((current) => ({
        ...current,
        [cameraId]: (current[cameraId] ?? []).filter((item) => item.id !== zone.id),
      }));
      if (editingZone?.id === zone.id) cancelZoneEdit();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete zone.");
    }
  }

  async function addPerson(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    try {
      const response = await fetch("/api/people", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ display_name: personName, category: personCategory }),
      });
      if (!response.ok) {
        const body = (await response.json()) as { detail?: string };
        throw new Error(body.detail ?? `Person registration failed (${response.status})`);
      }
      setPersonName("");
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not register person.");
    }
  }

  async function removePerson(person: PersonRecord) {
    setError("");
    try {
      const response = await fetch(`/api/people/${person.id}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error(`Person removal failed (${response.status})`);
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not remove person.");
    }
  }

  async function savePersonEdits(event: FormEvent<HTMLFormElement>, person: PersonRecord) {
    event.preventDefault();
    setError("");
    try {
      const response = await fetch(`/api/people/${person.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          display_name: editName,
          category: editCategory,
          active: person.active,
        }),
      });
      if (!response.ok) throw new Error(`Profile update failed (${response.status})`);
      setEditingPersonId(null);
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not update person.");
    }
  }

  async function setPersonActive(person: PersonRecord, active: boolean) {
    setError("");
    try {
      const response = await fetch(`/api/people/${person.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          display_name: person.display_name,
          category: person.category,
          active,
        }),
      });
      if (!response.ok) throw new Error(`Profile update failed (${response.status})`);
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not update person.");
    }
  }

  async function enrollPerson(event: FormEvent<HTMLFormElement>, personId: number) {
    event.preventDefault();
    setError("");
    const files = enrollmentFiles[personId] ?? [];
    if (files.length < 2 || files.length > 8) {
      setError("Choose between 2 and 8 enrollment photographs.");
      return;
    }
    if (!consentChecks[personId]) {
      setError("Confirm that the person gave informed consent before enrollment.");
      return;
    }
    const formData = new FormData();
    formData.set("consent_confirmed", "true");
    files.forEach((file) => formData.append("photos", file));
    try {
      const response = await fetch(`/api/people/${personId}/enroll`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = (await response.json()) as { detail?: string };
        throw new Error(body.detail ?? `Enrollment failed (${response.status})`);
      }
      setEnrollmentFiles((current) => ({ ...current, [personId]: [] }));
      setConsentChecks((current) => ({ ...current, [personId]: false }));
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not enroll person.");
    }
  }

  async function withdrawConsent(person: PersonRecord) {
    setError("");
    try {
      const response = await fetch(`/api/people/${person.id}/consent/withdraw`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`Consent withdrawal failed (${response.status})`);
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not withdraw consent.");
    }
  }

  async function saveForReview(event: DetectionEvent) {
    setError("");
    try {
      const response = await fetch("/api/review-events", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(event),
      });
      if (!response.ok) throw new Error(`Saving detection failed (${response.status})`);
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save detection.");
    }
  }

  async function markReviewed(eventId: string) {
    setError("");
    try {
      const selectedPerson = reviewSelections[eventId] ?? "";
      const response = await fetch(`/api/review-events/${eventId}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          reviewed: true,
          person_id: selectedPerson ? Number(selectedPerson) : null,
        }),
      });
      if (!response.ok) throw new Error(`Updating review failed (${response.status})`);
      await refreshPeopleData();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not update review.");
    }
  }

  const onlineCount = cameras.filter((camera) => camera.status === "online").length;
  const trackedCount = new Set(
    cameras.flatMap((camera) =>
      camera.detections.flatMap((detection) =>
        detection.track_id === null ? [] : [`${camera.id}:${detection.track_id}`],
      ),
    ),
  ).size;
  const describeDetection = (detection: Detection) => {
    if (detection.recognition_status.startsWith("VERIFIED_") && detection.recognized_name) {
      const category =
        detection.recognition_status === "VERIFIED_RESIDENT"
          ? "Resident"
          : detection.recognition_status === "VERIFIED_VISITOR"
            ? "Visitor"
            : "Staff";
      return `${detection.recognized_name} | ${category}`;
    }
    if (detection.predicted_class === "person") {
      if (detection.recognition_status === "IDENTIFYING") return "Identifying...";
      if (detection.recognition_status === "FACE_NOT_VISIBLE") {
        return "Person | Face Not Visible";
      }
      return "Unknown Person";
    }
    return detection.final_classification
      ? detection.final_classification
      : `unknown (model suggested ${detection.predicted_class}${
          detection.verification_alternative
            ? `; verifier suggested ${detection.verification_alternative}`
            : ""
        })`;
  };
  const categoryLabel = (category: string) =>
    ({
      resident: "Resident",
      visitor: "Visitor",
      authorized_staff: "Authorized staff",
    })[category] ?? category;
  const recognitionGuidance: Record<string, string> = {
    checking: "Checking local recognition setup...",
    ready: "Face matching is ready. Save a profile, then enroll 2–8 clear photos with the person's consent.",
    model_files_missing: "Face matching is unavailable: YuNet/SFace model files are missing. Add the configured ONNX files under backend/models and restart the backend. You can save profiles, but cannot enroll or identify people yet.",
    encryption_key_missing: "Face matching is unavailable: SENTINEL_BIOMETRIC_KEY is missing. Configure it in backend/.env and restart the backend. You can save profiles, but cannot enroll or identify people yet.",
    unsupported_device: "Face matching is unavailable: this installation supports CPU inference only.",
    model_load_failed: "Face matching could not load its local models. Check the backend logs and model files.",
    disabled: "Face matching is disabled in the backend configuration.",
  };
  const canEnrollFaces = recognitionStatus === "ready";

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">S</span>
          <span>Sentinel<span className="brand-accent">AI</span></span>
        </a>
        <div className="system-state">
          <span className={`pulse ${connected ? "live" : ""}`} />
          {connected ? "EVENT STREAM LIVE" : "CONNECTING"}
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">LOCAL SECURITY CONSOLE</p>
          <h1>Good to see you.</h1>
          <p className="subheading">Your cameras and live detections, in one place.</p>
        </div>
        <form className="add-camera" onSubmit={(event) => void addCamera(event)}>
          <input
            aria-label="Camera name"
            placeholder="Camera name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            required
            maxLength={80}
          />
          <input
            aria-label="Camera source"
            placeholder="Webcam index or stream URL"
            value={source}
            onChange={(event) => setSource(event.target.value)}
            required
          />
          <button type="submit">Add camera <span aria-hidden="true">+</span></button>
        </form>
      </section>

      {error && <div className="error-banner" role="alert">{error}</div>}

      <section className="stats" aria-label="Monitoring summary">
        <article className="stat-card">
          <span className="stat-icon">◉</span>
          <div><p>CAMERAS ONLINE</p><strong>{onlineCount}<small> / {cameras.length}</small></strong></div>
        </article>
        <article className="stat-card">
          <span className="stat-icon icon-green">⌖</span>
          <div><p>TRACKED OBJECTS</p><strong>{trackedCount.toString().padStart(2, "0")}</strong></div>
        </article>
        <article className="stat-card">
          <span className="stat-icon icon-amber">↗</span>
          <div><p>RECENT DETECTION UPDATES</p><strong>{events.length.toString().padStart(2, "0")}</strong></div>
        </article>
      </section>

      <div className="content-grid">
        <section>
          <div className="section-heading">
            <div><p className="eyebrow">LIVE MONITORING</p><h2>Camera feeds <span>{cameras.length}</span></h2></div>
            <span className="section-note">PROCESSING ON THIS DEVICE</span>
          </div>
          {cameras.length === 0 ? (
            <div className="empty-state">
              <span className="empty-icon">◉</span>
              <h3>No cameras connected</h3>
              <p>Add a webcam or IP camera stream to start local object detection.</p>
              <span className="hint">Try webcam index <code>0</code> or a camera stream URL.</span>
            </div>
          ) : (
            <div className="camera-grid">
              {cameras.map((camera) => (
                <article className="camera-card" key={camera.id}>
                  <div
                    className="video-frame"
                    style={{
                      aspectRatio: feedDimensions[camera.id]
                        ? `${feedDimensions[camera.id].width} / ${feedDimensions[camera.id].height}`
                        : undefined,
                    }}
                  >
                    {camera.timestamp ? (
                      <img
                        src={`/api/cameras/${camera.id}/stream.mjpg`}
                        alt={`${camera.name} live annotated feed`}
                        onLoad={(event) => {
                          const image = event.currentTarget;
                          setFeedDimensions((current) => {
                            const previous = current[camera.id];
                            if (
                              previous?.width === image.naturalWidth
                              && previous.height === image.naturalHeight
                            ) return current;
                            return {
                              ...current,
                              [camera.id]: {
                                width: image.naturalWidth,
                                height: image.naturalHeight,
                              },
                            };
                          });
                        }}
                      />
                    ) : (
                      <div className="feed-placeholder"><span className="spinner" /><p>Waiting for camera frames</p></div>
                    )}
                    <svg
                      className={`zone-overlay ${drawingCamera === camera.id ? "zone-overlay-active" : ""}`}
                      viewBox="0 0 1 1"
                      preserveAspectRatio="none"
                      aria-label={`${camera.name} detection zones`}
                      onPointerDown={addZonePoint}
                      onPointerMove={moveZoneVertex}
                      onPointerUp={() => setDraggedVertex(null)}
                      onPointerCancel={() => setDraggedVertex(null)}
                      style={{
                        pointerEvents:
                          drawingCamera === camera.id && camera.status === "online"
                            ? "auto"
                            : "none",
                      }}
                    >
                      {(zones[camera.id] ?? []).map((zone) => (
                        <g key={zone.id} className={zone.enabled ? "zone-shape" : "zone-shape disabled"}>
                          <polygon
                            points={zone.polygon.map((point) => `${point.x},${point.y}`).join(" ")}
                          />
                          <text x={zone.polygon[0]?.x ?? 0} y={zone.polygon[0]?.y ?? 0}>
                            {zone.name}
                          </text>
                        </g>
                      ))}
                      {drawingCamera === camera.id && draftPoints.length > 0 && (
                        <g className="zone-shape draft">
                          <polygon points={draftPoints.map((point) => `${point.x},${point.y}`).join(" ")} />
                          {draftPoints.map((point, index) => (
                            <circle
                              key={index}
                              cx={point.x}
                              cy={point.y}
                              r="0.012"
                              onPointerDown={(event) => {
                                if (editingZone) {
                                  event.stopPropagation();
                                  setDraggedVertex(index);
                                }
                              }}
                            />
                          ))}
                        </g>
                      )}
                    </svg>
                    <span className={`camera-badge ${camera.status}`}>{camera.status}</span>
                    {camera.timestamp && <time>{new Date(camera.timestamp).toLocaleTimeString()}</time>}
                  </div>
                  <div className="camera-details">
                    <div><h3>{camera.name}</h3><p>{camera.source_type === "webcam" ? "WEBCAM" : "NETWORK CAMERA"} · {camera.detections.length} detections</p></div>
                    <button className="remove-button" onClick={() => void removeCamera(camera)} aria-label={`Remove ${camera.name}`}>×</button>
                  </div>
                  {camera.error && <p className="camera-error">{camera.error}</p>}
                  <div className="zone-workspace">
                    <div className="zone-list-heading">
                      <strong>Detection zones</strong>
                      <button
                        className="small-action"
                        onClick={() => beginZoneDrawing(camera.id)}
                        disabled={!camera.timestamp || camera.status !== "online"}
                      >
                        Draw zone
                      </button>
                    </div>
                    {(zones[camera.id] ?? []).map((zone) => (
                      <div className="zone-row" key={zone.id}>
                        <span><strong>{zone.name}</strong><small>{zone.loiter_seconds}s loiter · {zone.enabled ? "enabled" : "disabled"}</small></span>
                        <div>
                          <button className="small-action" onClick={() => void toggleZone(camera.id, zone)}>
                            {zone.enabled ? "Disable" : "Enable"}
                          </button>
                          <button className="small-action" onClick={() => beginZoneEdit(camera.id, zone)}>Edit</button>
                          <button className="small-action danger-action" onClick={() => void deleteZone(camera.id, zone)}>Delete</button>
                        </div>
                      </div>
                    ))}
                    {drawingCamera === camera.id && (
                      <form
                        className="zone-editor"
                        onSubmit={(event) => {
                          event.preventDefault();
                          void saveZone(camera.id);
                        }}
                      >
                        <p>{editingZone ? "Drag the points to reshape this polygon." : "Click the live feed to add polygon points."} Add at least 3 points.</p>
                        <label>
                          Zone name
                          <input value={zoneName} onChange={(event) => setZoneName(event.target.value)} required maxLength={80} />
                        </label>
                        <div className="zone-settings">
                          <label>Loiter (seconds)<input type="number" min="1" max="86400" value={zoneLoiterSeconds} onChange={(event) => setZoneLoiterSeconds(Number(event.target.value))} required /></label>
                          <label>Cooldown (seconds)<input type="number" min="0" max="86400" value={zoneCooldownSeconds} onChange={(event) => setZoneCooldownSeconds(Number(event.target.value))} required /></label>
                        </div>
                        <div className="zone-editor-actions">
                          <button type="submit">{editingZone ? "Save changes" : "Create zone"}</button>
                          <button type="button" className="small-action" onClick={cancelZoneEdit}>Cancel</button>
                          <button type="button" className="small-action" onClick={() => setDraftPoints((current) => current.slice(0, -1))} disabled={draftPoints.length === 0}>Undo point</button>
                        </div>
                      </form>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <aside className="activity-panel">
          <div className="section-heading">
            <div><p className="eyebrow">LIVE DETECTIONS</p><h2>Detection updates</h2></div>
            <span className={`live-label ${connected ? "" : "disconnected"}`}><i />{connected ? "LIVE" : "OFFLINE"}</span>
          </div>
          <div className="event-list">
            {zoneEvents.slice(0, 8).map((event, index) => (
              <article className="event-row zone-event-row" key={`${event.camera_id}-${event.timestamp}-${index}`}>
                <span className="event-dot" />
                <div className="event-copy">
                  <strong>{event.event_type === "loitering" ? "Loitering" : `Zone ${event.event_type}`}</strong>
                  <p>{event.camera_name} · {event.zone_name} · {event.object_label}</p>
                  <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
                </div>
              </article>
            ))}
            {events.length === 0 ? (
              zoneEvents.length === 0 && <div className="activity-empty"><span>⌁</span><p>Live object observations and zone-rule events will appear here when a camera is active.</p></div>
            ) : events.slice(0, 18).map((event, index) => (
              <article className="event-row" key={`${event.camera_id}-${event.timestamp}-${index}`}>
                <span className="event-dot" />
                <div className="event-copy">
                  <strong>{event.detections.length} object{event.detections.length === 1 ? "" : "s"} detected</strong>
                  <p>{event.camera_name}{event.detections.length ? ` · ${event.detections.map(describeDetection).join(", ")}` : ""}</p>
                  <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
                  {event.detections.length > 0 && (
                    <button className="small-action" onClick={() => void saveForReview(event)}>
                      Save for review
                    </button>
                  )}
                </div>
              </article>
            ))}
          </div>
          <p className="activity-footnote">Unknown means the available evidence was insufficient. Model confidence is not a calibrated probability; detections are not proof of wrongdoing.</p>
        </aside>
      </div>

      <section className="people-workspace">
        <div className="section-heading">
          <div><p className="eyebrow">PEOPLE MANAGEMENT</p><h2>Registered people</h2></div>
          <span className={`section-note recognition-state ${recognitionStatus === "ready" ? "ready" : ""}`}>
            FACE RECOGNITION · {recognitionStatus.replace(/_/g, " ")}
          </span>
        </div>
        <div className={`recognition-guidance ${canEnrollFaces ? "ready" : "unavailable"}`} role="status">
          <strong>{canEnrollFaces ? "Enrollment available" : "Enrollment unavailable"}</strong>
          <p>{recognitionGuidance[recognitionStatus] ?? `Face recognition is unavailable (${recognitionStatus}). Check backend configuration.`}</p>
        </div>
        <div className="people-grid">
          <article className="data-card">
            <h3>Register a person</h3>
            <form className="person-form" onSubmit={(event) => void addPerson(event)}>
              <input aria-label="Display name" placeholder="Display name" value={personName} onChange={(event) => setPersonName(event.target.value)} required maxLength={100} />
              <select aria-label="Person category" value={personCategory} onChange={(event) => setPersonCategory(event.target.value as PersonRecord["category"])}>
                <option value="resident">Resident</option>
                <option value="visitor">Visitor</option>
                <option value="authorized_staff">Authorized staff</option>
              </select>
              <button type="submit">Add person</button>
            </form>
            <p className="registration-hint">This saves the person's name and category. To enable recognition, enroll their consented face photos in the profile after saving.</p>
            <div className="record-list">
              {people.length === 0 ? <p className="record-empty">No person records yet.</p> : people.map((person) => (
                <div className="person-entry" key={person.id}>
                  <div className="person-record">
                    <span>
                      <strong>{person.display_name}</strong>
                      <small>{categoryLabel(person.category)} · {person.active ? "Active" : "Inactive"} · {person.enrollment_count} enrolled sample{person.enrollment_count === 1 ? "" : "s"}</small>
                      <small>{person.enrollment_count > 0 && person.consented_at ? "Face enrolled · consent recorded" : "Face not enrolled"}</small>
                    </span>
                    <button className="remove-button" onClick={() => void removePerson(person)} aria-label={`Delete ${person.display_name}`}>×</button>
                  </div>
                  <div className="person-actions">
                    <button className="small-action" onClick={() => {
                      setEditingPersonId(person.id);
                      setEditName(person.display_name);
                      setEditCategory(person.category);
                    }}>Edit</button>
                    <button className="small-action" onClick={() => void setPersonActive(person, !person.active)}>
                      {person.active ? "Deactivate" : "Activate"}
                    </button>
                    {person.consented_at && (
                      <button className="small-action danger-action" onClick={() => void withdrawConsent(person)}>
                        Withdraw consent
                      </button>
                    )}
                  </div>
                  {editingPersonId === person.id && (
                    <form className="profile-edit" onSubmit={(event) => void savePersonEdits(event, person)}>
                      <input aria-label="Edit display name" value={editName} onChange={(event) => setEditName(event.target.value)} required maxLength={100} />
                      <select aria-label="Edit category" value={editCategory} onChange={(event) => setEditCategory(event.target.value as PersonRecord["category"])}>
                        <option value="resident">Resident</option>
                        <option value="visitor">Visitor</option>
                        <option value="authorized_staff">Authorized staff</option>
                      </select>
                      <button className="small-action" type="submit">Save changes</button>
                      <button className="small-action" type="button" onClick={() => setEditingPersonId(null)}>Cancel</button>
                    </form>
                  )}
                  {person.active && (
                    <form className="enrollment-form" onSubmit={(event) => void enrollPerson(event, person.id)}>
                      <label className="file-picker">
                        Enrollment photos (2–8)
                        <input
                          type="file"
                          accept="image/jpeg,image/png,image/webp"
                          multiple
                          disabled={!canEnrollFaces}
                          onChange={(event) => setEnrollmentFiles((current) => ({
                            ...current,
                            [person.id]: Array.from(event.target.files ?? []),
                          }))}
                        />
                      </label>
                      <small>{(enrollmentFiles[person.id] ?? []).length} photo(s) selected. Files are processed locally and discarded after enrollment.</small>
                      <label className="consent-check">
                        <input
                          type="checkbox"
                          disabled={!canEnrollFaces}
                          checked={consentChecks[person.id] ?? false}
                          onChange={(event) => setConsentChecks((current) => ({ ...current, [person.id]: event.target.checked }))}
                        />
                        I confirm this person gave informed consent to local face enrollment.
                      </label>
                      <button className="small-action" type="submit" disabled={!canEnrollFaces}>Enroll face samples</button>
                    </form>
                  )}
                </div>
              ))}
            </div>
          </article>
          <article className="data-card">
            <h3>Saved detection reviews <span>{reviewEvents.length}</span></h3>
            {reviewEvents.length === 0 ? (
              <p className="record-empty">Use “Save for review” on a live detection update to add one here.</p>
            ) : (
              <div className="record-list review-list">
                {reviewEvents.map((reviewEvent) => (
                  <div className="review-record" key={reviewEvent.id}>
                    <div className="review-summary">
                      <strong>{reviewEvent.camera_name} · {reviewEvent.detections.map(describeDetection).join(", ")}</strong>
                      <small>{new Date(reviewEvent.timestamp).toLocaleString()}</small>
                      {reviewEvent.reviewed && (
                        <small className="reviewed-label">
                          Reviewed{reviewEvent.person_name ? ` · ${reviewEvent.person_name} (${reviewEvent.person_role})` : " · no person associated"}
                        </small>
                      )}
                    </div>
                    {!reviewEvent.reviewed && (
                      <div className="review-controls">
                        <select
                          aria-label={`Associate a person with detection from ${reviewEvent.camera_name}`}
                          value={reviewSelections[reviewEvent.id] ?? ""}
                          onChange={(event) => setReviewSelections((current) => ({ ...current, [reviewEvent.id]: event.target.value }))}
                        >
                          <option value="">No person / unidentified</option>
                          {people.map((person) => <option key={person.id} value={person.id}>{person.display_name} · {person.role}</option>)}
                        </select>
                        <button className="small-action" onClick={() => void markReviewed(reviewEvent.id)}>Mark reviewed</button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </article>
        </div>
        <p className="activity-footnote">Only verified matches display a registered name. Similarity is not a probability; unknown or low-quality faces are never treated as suspicious. Enrollment images are not retained; encrypted embeddings remain local.</p>
      </section>

      <section className="people-workspace">
        <div className="section-heading">
          <div><p className="eyebrow">IDENTITY AUDIT</p><h2>Recognition history <span>{recognitionEvents.length}</span></h2></div>
          <span className="section-note">VERIFICATION RESULTS ONLY</span>
        </div>
        <div className="data-card">
          {recognitionEvents.length === 0 ? (
            <p className="record-empty">Recognition results will appear here after configured models process tracked people.</p>
          ) : (
            <div className="record-list">
              {recognitionEvents.map((item) => (
                <div className="recognition-record" key={item.id}>
                  <strong>{item.person_name ?? item.status.replace(/_/g, " ")}</strong>
                  <small>{item.person_category ? categoryLabel(item.person_category) : "No verified identity"} · camera {item.camera_id} · track {item.tracking_id}</small>
                  <time>{new Date(item.timestamp).toLocaleString()}</time>
                  {item.similarity !== null && <small>Similarity score: {item.similarity.toFixed(3)} (not a probability)</small>}
                </div>
              ))}
            </div>
          )}
        </div>
        <p className="activity-footnote">Recognition history is stored locally. Deleting a profile deletes its encrypted enrollment samples; recorded events remain but lose the linked profile.</p>
      </section>

      <footer><span>SentinelAI <span className="brand-accent">/</span> PHASE 1</span><span>LOCAL-FIRST · NO CONTINUOUS RECORDING</span></footer>
    </main>
  );
}

export default App;
