fetch("data/dashboard-data.json")
  .then((response) => response.ok ? response.json() : null)
  .then((payload) => {
    if (!payload) return;
    document.querySelectorAll(".source-line").forEach((el) => {
      el.textContent = `Static data snapshot: ${payload.scope.players} players, ${payload.scope.historyRows} checked player-GW rows, ${payload.scope.forecastRows} forecast rows, across ${payload.scope.teams.length} clubs.`;
    });
  })
  .catch(() => {});
