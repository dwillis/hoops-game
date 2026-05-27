"""``hoops`` CLI entrypoint. Phase 5 ships the ``play`` subcommand only;
``simulate-season`` and ``bracket`` arrive in Phase 8.
"""

from __future__ import annotations

import polars as pl
import typer

from hoops.data.distributions import load_league_prior, load_team_priors
from hoops.data.paths import fitted_seasons, teams_path
from hoops.data.rosters import load_roster
from hoops.engine.attribution import attribute_players
from hoops.engine.machine import simulate_game
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.league import League
from hoops.rules import rules_for

app = typer.Typer(help="Hoops 2026: WBB coaching simulator.")


def _resolve_team(query: str, season: str) -> tuple[int, str]:
    """Resolve a user-friendly team string to (team_id, display_name).

    Match priority (first non-empty wins):
    1. Exact team_slug match.
    2. Exact team_name match (case-insensitive; hyphens treated as spaces).
    3. Substring match on team_slug.
    4. Substring match on team_name.
    Ambiguous matches produce a BadParameter listing candidates.
    """
    df = pl.read_parquet(teams_path(League.WBB, season))
    q = query.lower()
    q_as_name = q.replace("-", " ")

    exact_slug = df.filter(pl.col("team_slug").str.to_lowercase() == q)
    if exact_slug.height == 1:
        row = exact_slug.row(0, named=True)
        return int(row["team_id"]), row["team_name"]

    exact_name = df.filter(pl.col("team_name").str.to_lowercase() == q_as_name)
    if exact_name.height == 1:
        row = exact_name.row(0, named=True)
        return int(row["team_id"]), row["team_name"]

    slug_sub = df.filter(pl.col("team_slug").str.to_lowercase().str.contains(q))
    if slug_sub.height == 1:
        row = slug_sub.row(0, named=True)
        return int(row["team_id"]), row["team_name"]

    name_sub = df.filter(pl.col("team_name").str.to_lowercase().str.contains(q_as_name))
    if name_sub.height == 1:
        row = name_sub.row(0, named=True)
        return int(row["team_id"]), row["team_name"]

    matches = pl.concat([slug_sub, name_sub]).unique("team_id")
    if matches.height == 0:
        raise typer.BadParameter(f"no team matching {query!r} in {season}")
    candidates = ", ".join(matches.head(8)["team_name"].to_list())
    raise typer.BadParameter(
        f"{query!r} is ambiguous in {season}; candidates include: {candidates}. "
        "Pass a more specific query (e.g. the full team slug)."
    )


@app.command()
def seasons() -> None:
    """List seasons with fitted data ready to play."""
    available = fitted_seasons(League.WBB)
    if not available:
        typer.echo("No fitted seasons found. Run scripts/fit_distributions.py first.")
        raise typer.Exit(1)
    for s in available:
        typer.echo(s)


@app.command()
def play(
    season: str = typer.Option(None, "--season", help="Season for both teams (auto-detected if omitted)"),
    home: str = typer.Option(None, "--home", help="Home team name or slug fragment (skips the picker)"),
    away: str = typer.Option(None, "--away", help="Away team name or slug fragment (skips the picker)"),
    home_season: str = typer.Option(None, "--home-season", help="Season for home team (overrides --season)"),
    away_season: str = typer.Option(None, "--away-season", help="Season for away team (overrides --season)"),
    seed: int = typer.Option(None, "--seed", help="RNG seed (random if omitted)"),
    all_teams: bool = typer.Option(
        False, "--all-teams",
        help="Include sub-D-I teams in the picker (default: D-I only)",
    ),
    neutral: bool = typer.Option(
        False, "--neutral",
        help="Neutral site — no home court advantage",
    ),
) -> None:
    """Open the Textual UI.

    With no ``--home`` / ``--away`` flags, an in-app team picker is shown.
    With both flags, the picker is skipped and the game runs directly.
    """
    # Lazy import so plain `hoops --help` doesn't pull Textual.
    from hoops.ui.app import HoopsApp

    season_explicit = season is not None
    available = fitted_seasons(League.WBB)

    if season is None:
        if len(available) == 1:
            season = available[0]
            season_explicit = True
        elif len(available) == 0:
            typer.echo("No fitted seasons found. Run scripts/fit_distributions.py first.")
            raise typer.Exit(1)

    # Resolve per-side seasons: --home-season / --away-season override --season.
    h_season = home_season or season
    a_season = away_season or season

    if home is not None or away is not None:
        if h_season is None or a_season is None:
            typer.echo(
                "Multiple seasons available; pass --season (or --home-season / --away-season) "
                f"when using --home/--away. Options: {', '.join(available)}"
            )
            raise typer.Exit(1)

    if home is None and away is None:
        HoopsApp(
            season=season,
            seed=seed,
            division_one_only=not all_teams,
            season_explicit=season_explicit,
            neutral_site=neutral,
        ).run()
        return

    if home is None or away is None:
        raise typer.BadParameter(
            "either pass both --home and --away (skip the picker), or neither (use it)"
        )

    home_id, home_name = _resolve_team(home, h_season)
    away_id, away_name = _resolve_team(away, a_season)
    if home_id == away_id and h_season == a_season:
        raise typer.BadParameter("home and away resolved to the same team in the same season")

    home_priors_all = {p.team_id: p for p in load_team_priors(League.WBB, h_season)}
    away_priors_all = {p.team_id: p for p in load_team_priors(League.WBB, a_season)}
    league_prior = load_league_prior(League.WBB, max(h_season, a_season))
    rules = rules_for(League.WBB, max(h_season, a_season))

    home_roster = load_roster(home_id, h_season)
    away_roster = load_roster(away_id, a_season)
    rng = make_rng(seed=seed)
    final, events = simulate_game(
        home_priors_all[home_id], away_priors_all[away_id], rules,
        rng, Side.HOME, league=league_prior,
        home_roster=home_roster, away_roster=away_roster,
        neutral_site=neutral,
    )
    events = attribute_players(events, home_roster, away_roster, rng)
    typer.echo(
        f"Simulated {home_name} {final.home_score} – {final.away_score} {away_name} "
        f"(seed={seed}); {len(events)} events. Opening UI…"
    )

    HoopsApp(events=events, home_name=home_name, away_name=away_name).run()


@app.command()
def bracket(
    season: str = typer.Option(None, "--season", help="Season bracket to play"),
    team: str = typer.Option(None, "--team", help="Team to coach (name or slug)"),
    seed: int = typer.Option(42, "--seed", help="RNG seed"),
) -> None:
    """Play an NCAA tournament bracket. Coach one team through March Madness.

    Uses historical bracket data extracted from schedule data. Other games
    are auto-simulated; you coach your team's games interactively.
    """
    from hoops.data.paths import bracket_seasons, bracket_path
    from hoops.engine.bracket import Bracket

    available = bracket_seasons(League.WBB)
    if not available:
        typer.echo("No bracket data found. Run scripts/extract_brackets.py first.")
        raise typer.Exit(1)

    if season is None:
        if len(available) == 1:
            season = available[0]
        else:
            typer.echo(
                f"Multiple bracket seasons available: {', '.join(available)}\n"
                "Pass --season to choose one."
            )
            raise typer.Exit(1)

    if season not in available:
        typer.echo(f"No bracket data for {season}. Available: {', '.join(available)}")
        raise typer.Exit(1)

    if team is None:
        bp = bracket_path(League.WBB, season)
        b = Bracket.load(bp)
        bracket_team_ids = b.team_ids()
        priors = {p.team_id: p for p in load_team_priors(League.WBB, season)}
        bracket_teams = sorted(
            [(tid, priors[tid].team_name) for tid in bracket_team_ids if tid in priors],
            key=lambda t: t[1],
        )
        typer.echo(f"Tournament teams for {season}:")
        for tid, name in bracket_teams:
            typer.echo(f"  {name}")
        typer.echo("\nPass --team <name> to choose your team.")
        raise typer.Exit(0)

    team_id, team_name = _resolve_team(team, season)

    bp = bracket_path(League.WBB, season)
    b = Bracket.load(bp)
    if team_id not in b.team_ids():
        typer.echo(f"{team_name} (id={team_id}) is not in the {season} NCAA tournament bracket.")
        raise typer.Exit(1)

    from hoops.ui.tournament_app import TournamentApp

    typer.echo(f"Starting {season} NCAA Tournament as {team_name} (seed={seed})...")
    TournamentApp(
        season=season,
        user_team_id=team_id,
        user_team_name=team_name,
        seed=seed,
    ).run()


@app.command()
def conference(
    season: str = typer.Option(None, "--season", help="Season to play"),
    conf: str = typer.Option(None, "--conf", help="Conference name or fragment (e.g. 'SEC', 'Big Ten')"),
    team: str = typer.Option(None, "--team", help="Team to coach (name or slug)"),
    seed: int = typer.Option(42, "--seed", help="RNG seed"),
) -> None:
    """Play a conference tournament. Coach one team through their conference tourney.

    Uses historical bracket data extracted from schedule data. Other games
    are auto-simulated; you coach your team's games interactively.
    """
    from hoops.data.paths import list_conf_tournaments, conf_tournament_path

    # Resolve season
    available = fitted_seasons(League.WBB)
    if not available:
        typer.echo("No fitted seasons found.")
        raise typer.Exit(1)

    if season is None:
        if len(available) == 1:
            season = available[0]
        else:
            typer.echo(f"Multiple seasons available: {', '.join(available)}\nPass --season to choose one.")
            raise typer.Exit(1)

    # List available conferences
    conferences = list_conf_tournaments(League.WBB, season)
    if not conferences:
        typer.echo(f"No conference tournament data for {season}. Run scripts/extract_conf_tournaments.py first.")
        raise typer.Exit(1)

    if conf is None:
        typer.echo(f"Conference tournaments for {season}:")
        for c in sorted(conferences, key=lambda c: c["conference_name"]):
            typer.echo(f"  {c['conference_name']} ({c['num_teams']} teams, {c['num_games']} games)")
        typer.echo("\nPass --conf <name> to choose a conference.")
        raise typer.Exit(0)

    # Resolve conference by name fragment
    conf_lower = conf.lower()
    matches = [c for c in conferences if conf_lower in c["conference_name"].lower()]
    if len(matches) == 0:
        typer.echo(f"No conference matching {conf!r}. Available:")
        for c in sorted(conferences, key=lambda c: c["conference_name"]):
            typer.echo(f"  {c['conference_name']}")
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(f"Ambiguous: {conf!r} matches multiple conferences:")
        for c in matches:
            typer.echo(f"  {c['conference_name']}")
        raise typer.Exit(1)

    conf_info = matches[0]
    tid = conf_info["tournament_id"]
    conf_name = conf_info["conference_name"]

    # Load bracket to list teams or validate team
    from hoops.engine.bracket import Bracket
    bp = conf_tournament_path(League.WBB, season, tid)
    b = Bracket.load(bp)
    bracket_tids = b.team_ids()

    if team is None:
        priors = {p.team_id: p for p in load_team_priors(League.WBB, season)}
        bracket_teams = sorted(
            [(t, priors[t].team_name) for t in bracket_tids if t in priors],
            key=lambda t: t[1],
        )
        typer.echo(f"{conf_name} teams for {season}:")
        for t_id, name in bracket_teams:
            typer.echo(f"  {name}")
        typer.echo("\nPass --team <name> to choose your team.")
        raise typer.Exit(0)

    team_id, team_name = _resolve_team(team, season)
    if team_id not in bracket_tids:
        typer.echo(f"{team_name} is not in the {season} {conf_name}.")
        raise typer.Exit(1)

    from hoops.ui.tournament_app import TournamentApp
    typer.echo(f"Starting {season} {conf_name} as {team_name} (seed={seed})...")
    TournamentApp(
        season=season,
        user_team_id=team_id,
        user_team_name=team_name,
        seed=seed,
        bracket_path_override=bp,
    ).run()


@app.command()
def version() -> None:
    """Print the installed hoops version."""
    from hoops import __version__
    typer.echo(__version__)


if __name__ == "__main__":
    app()
