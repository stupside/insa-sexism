import typer

app = typer.Typer(help="This is a CLI tool detect sexism in text.", no_args_is_help=True)

@app.command()
def ping():
    typer.echo("Pong")

@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")