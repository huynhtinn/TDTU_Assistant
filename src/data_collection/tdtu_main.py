
import os
import click
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from tdtu_client import TDTUClient
from tdtu_db import open_tdtu_db, QuyCheDocument
import json

load_dotenv()

@click.group()
def cli():
    pass

@click.command()
@click.option('--headless/--no-headless', default=True, help='Run browser in headless mode')
def crawl_quiche(headless):
    logger.info("Crawling TDTU Quy Che documents...")
    
    username = os.getenv("TDTU_USERNAME")
    password = os.getenv("TDTU_PASSWORD")
    
    if not username or not password:
        logger.error("TDTU_USERNAME and TDTU_PASSWORD not found in .env file")
        click.echo("❌ Please add TDTU_USERNAME and TDTU_PASSWORD to your .env file")
        return
    
    # Initialize TDTU database
    tdtu_eng = open_tdtu_db()
    
    try:
        with TDTUClient(headless=headless, verbose=True) as client:
            # Login
            if not client.login(username, password):
                logger.error("Failed to login to TDTU portal")
                click.echo("❌ Login failed! Please check your credentials.")
                return
            
            click.echo("✓ Login successful!")
            
            # Extract quy che list
            quy_che_list = client.extract_quiche_list()
            
            if not quy_che_list:
                logger.warning("No documents found")
                click.echo("⚠ No documents found. The page structure may have changed.")
                
                return
            
            click.echo(f"✓ Found {len(quy_che_list)} documents")
            
            # Save to JSON file FIRST (before database)
            with open('tdtu_quy_che_list.json', 'w', encoding='utf-8') as f:
                json.dump(quy_che_list, f, indent=2, ensure_ascii=False)
            
            click.echo("✓ Exported to tdtu_quy_che_list.json")
            
            # Save to database
            with Session(tdtu_eng) as sess:
                for doc_data in quy_che_list:
                    doc = QuyCheDocument.from_json(doc_data)
                    QuyCheDocument.save(sess, doc)
                    click.echo(f"  → {doc.title}")
                
                sess.commit()
            
            click.echo(f"\n✓ Saved {len(quy_che_list)} documents to database")
            
            # Save to JSON file
            with open('tdtu_quy_che_list.json', 'w', encoding='utf-8') as f:
                json.dump(quy_che_list, f, indent=2, ensure_ascii=False)
            
            click.echo("✓ Exported to tdtu_quy_che_list.json")
            
    except Exception as e:
        logger.error(f"Error crawling TDTU: {e}")
        click.echo(f"❌ Error: {e}")


# Register commands
cli.add_command(crawl_quiche)

if __name__ == "__main__":
    logger.add("tdtu_crawler_{time}.log")
    cli()
