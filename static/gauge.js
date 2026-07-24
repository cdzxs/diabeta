function drawGauge(canvasId, probability) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const cx = canvas.width / 2;
    const cy = canvas.height - 20;
    const radius = canvas.width / 2 - 26;

    function colorForProb(p) {
        if (p < 0.3) return "#1fa971";
        if (p < 0.7) return "#f0b429";
        return "#e45858";
    }

    function drawArc(progress) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
        ctx.strokeStyle = "#e6edf2";
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, Math.PI + progress * Math.PI);
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
        ctx.strokeStyle = colorForProb(probability);
        ctx.stroke();

        ctx.fillStyle = "#0e2f56";
        ctx.font = "800 32px Segoe UI, Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText((probability * 100).toFixed(1) + "%", cx, cy - 34);

        ctx.fillStyle = "#607084";
        ctx.font = "700 13px Segoe UI, Arial, sans-serif";
        ctx.fillText("high-risk probability", cx, cy - 8);
    }

    let current = 0;
    const target = Math.max(0, Math.min(1, probability));

    function animate() {
        current += Math.max(0.01, (target - current) * 0.08);
        if (current > target || target - current < 0.005) current = target;
        drawArc(current);

        if (current < target) {
            requestAnimationFrame(animate);
        }
    }

    animate();
}
