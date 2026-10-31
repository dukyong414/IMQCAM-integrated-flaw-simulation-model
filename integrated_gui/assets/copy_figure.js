// Copy the Plotly graph as it is on screen right now (current scroll-zoom and rotation).
// Title and view buttons are cropped off; the legend stays. The camera is not changed.
function _liveSceneCamera(gd) {
    const scene = gd._fullLayout && gd._fullLayout.scene;
    const cam = (scene && scene.camera)
        || (gd.layout && gd.layout.scene && gd.layout.scene.camera);
    if (!cam || !cam.eye) {
        return null;
    }
    return JSON.parse(JSON.stringify(cam));
}

function _cropTop(blob, topPx) {
    if (!topPx || topPx < 2) {
        return Promise.resolve(blob);
    }
    return createImageBitmap(blob).then(function (bmp) {
        const canvas = document.createElement("canvas");
        const height = Math.max(1, bmp.height - topPx);
        canvas.width = bmp.width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(bmp, 0, topPx, bmp.width, height, 0, 0, bmp.width, height);
        return new Promise(function (resolve) {
            canvas.toBlob(function (out) { resolve(out || blob); }, "image/png");
        });
    });
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    figtools: {
        copy: async function (nClicks, graphId) {
            if (!nClicks) {
                return window.dash_clientside.no_update;
            }
            const host = document.getElementById(graphId);
            const gd = host && (host.classList.contains("js-plotly-plot")
                ? host
                : host.querySelector(".js-plotly-plot"));
            if (!gd || !window.Plotly) {
                return "No figure to copy yet.";
            }
            if (!navigator.clipboard || !window.ClipboardItem) {
                return "This browser does not allow copying images; use the camera icon to download a PNG.";
            }
            try {
                // ClipboardItem must be created inside the click. The promise records
                // the camera that is on screen after the latest scroll-zoom.
                const blob = (async () => {
                    const cam = _liveSceneCamera(gd);
                    if (cam) {
                        gd.layout.scene = gd.layout.scene || {};
                        // toImage reads gd.layout; scroll-zoom lives on the live scene camera.
                        gd.layout.scene.camera = cam;
                    }
                    const width = (gd._fullLayout && gd._fullLayout.width) || gd.clientWidth;
                    const height = (gd._fullLayout && gd._fullLayout.height) || gd.clientHeight;
                    const scale = 2;
                    const url = await window.Plotly.toImage(gd, {
                        format: "png",
                        width: width,
                        height: height,
                        scale: scale,
                    });
                    const raw = await (await fetch(url)).blob();
                    const marginT = (gd._fullLayout && gd._fullLayout.margin && gd._fullLayout.margin.t) || 0;
                    // Drop the title / button band. The 3D camera and legend are unchanged.
                    return _cropTop(raw, Math.round(marginT * scale));
                })();
                await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                return "Image copied to clipboard (" + new Date().toLocaleTimeString() + ")";
            } catch (err) {
                return "Copy failed: " + err;
            }
        },
        // Full chart, including subplot titles and the legend. No camera change, no crop.
        copyChart: async function (nClicks, graphId) {
            if (!nClicks) {
                return window.dash_clientside.no_update;
            }
            const host = document.getElementById(graphId);
            const gd = host && (host.classList.contains("js-plotly-plot")
                ? host
                : host.querySelector(".js-plotly-plot"));
            if (!gd || !window.Plotly) {
                return "No figure to copy yet.";
            }
            if (!navigator.clipboard || !window.ClipboardItem) {
                return "This browser does not allow copying images; use the camera icon to download a PNG.";
            }
            try {
                const blob = (async () => {
                    const width = (gd._fullLayout && gd._fullLayout.width) || gd.clientWidth;
                    const height = (gd._fullLayout && gd._fullLayout.height) || gd.clientHeight;
                    const url = await window.Plotly.toImage(gd, {
                        format: "png",
                        width: width,
                        height: height,
                        scale: 2,
                    });
                    return (await fetch(url)).blob();
                })();
                await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                return "Image copied to clipboard (" + new Date().toLocaleTimeString() + ")";
            } catch (err) {
                return "Copy failed: " + err;
            }
        },
    },
});
