fetch("data/dashboard-data.json")
  .then((response) => response.ok ? response.json() : null)
  .then((payload) => {
    if (!payload) return;
    document.querySelectorAll(".source-line").forEach((el, index) => {
      if (index === 0) {
        el.textContent = `Static data snapshot: ${payload.scope.players} players across ${payload.scope.teams.length} clubs.`;
      }
    });
  })
  .catch(() => {});

