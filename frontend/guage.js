function drawGauge(canvasId, probability) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    const cx = canvas.width / 2;
    const cy = canvas.height - 10;
    const radius = canvas.width / 2 - 20;

    function colorForProb(p) {
        if (p < 0.3) return "#2ecc71";
        if (p < 0.7) return "#f1c40f";
        return "#e74c3c";
    }

    // background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 18;
    ctx.strokeStyle = "#1f222e";
    ctx.stroke();

    // animated fill arc
    let current = 0;
    const target = probability;
    const color = colorForProb(target);

    function animate() {
        current += 0.02;
        if (current > target) current = target;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
        ctx.lineWidth = 18;
        ctx.strokeStyle = "#1f222e";
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, Math.PI + current * Math.PI);
        ctx.lineWidth = 18;
        ctx.strokeStyle = color;
        ctx.lineCap = "round";
        ctx.stroke();

        ctx.fillStyle = "#e6e6e6";
        ctx.font = "bold 22px Segoe UI";
        ctx.textAlign = "center";
        ctx.fillText((target * 100).toFixed(1) + "%", cx, cy - 20);

        if (current < target) {
            requestAnimationFrame(animate);
        }
    }
    animate();
}