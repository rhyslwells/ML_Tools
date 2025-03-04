import click
import json
import pprint

@click.group()
@click.pass_context
@click.argument("document")
def cli(ctx, document):
    """CLI for interacting with JSON documents"""
    with open(document) as file:
        ctx.obj = json.load(file)

@cli.command()
@click.pass_context
def show_keys(ctx):
    """Show all keys in the document"""
    click.echo("Keys: " + str(list(ctx.obj.keys())))

@cli.command()
@click.argument("key")
@click.pass_context
def get_value(ctx, key):
    """Retrieve a value by key"""
    value = ctx.obj.get(key, "Key not found")
    click.echo(value)

@cli.command()
@click.option("-k", "--key", required=True, help="Key to update")
@click.option("-v", "--value", required=True, help="New value")
@click.pass_context
def update_value(ctx, key, value):
    """Update a key's value"""
    ctx.obj[key] = value
    with open(ctx.parent.params['document'], 'w') as file:
        json.dump(ctx.obj, file, indent=4)
    click.echo("Updated successfully.")

if __name__ == '__main__':
    cli()