import io
import base64

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

from src.cool_poker import WinProba

wp = WinProba()

def calculate_probabilities(my_cards, table_cards, opponent_cards, num_opponents, num_simulations):
    try:
        # Clean and prepare inputs
        my_cards = [card.strip() for card in my_cards.split(',')]
        table_cards_list = [card.strip() if card.strip() else '' for card in table_cards.split(',')]
        
        # Ensure table_cards has 5 positions
        while len(table_cards_list) < 5:
            table_cards_list.append('')
        
        # Process opponent cards
        opponents_cards = []
        if opponent_cards.strip():
            opponent_pairs = opponent_cards.split(';')
            for pair in opponent_pairs:
                cards = [card.strip() for card in pair.split(',')]
                if len(cards) == 2:
                    opponents_cards.append(cards)
        
        # Calculate probabilities
        win_prob, tie_prob, loss_prob, combo_stats = wp.calculate_win_probability(
            my_cards, table_cards_list, num_opponents, opponents_cards, num_simulations
        )
        
        # Create plot for combination statistics
        buf = io.BytesIO()
        fig = plt.figure(figsize=(10, 6))
        wp.plot_combinations_statistics(combo_stats, show=False)
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Prepare results
        results = f"Win Probability: {win_prob:.2%}\n"
        results += f"Tie Probability: {tie_prob:.2%}\n"
        results += f"Loss Probability: {loss_prob:.2%}"

        plot_image = Image.open(buf)
        
        return results, plot_image
    
    except ValueError as e:
        # Return the error message without a plot
        empty_image = Image.new('RGB', (10, 10), color = 'white')
        return f"Error: {str(e)}", empty_image
    
    except Exception as e:
        # Catch any other unexpected errors
        empty_image = Image.new('RGB', (10, 10), color = 'white')
        return f"An unexpected error occurred: {str(e)}", empty_image

# Create Gradio interface
with gr.Blocks(title="Poker Win Probability Calculator") as demo:
    gr.Markdown("# Poker Win Probability Calculator")
    
    # Input section
    with gr.Row():
        with gr.Column():
            my_cards_input = gr.Textbox(
                label="Your Cards (comma separated, e.g., 'Kh, 9d')",
                placeholder="Required to fill out"            
            )
            table_cards_input = gr.Textbox(
                label="Table Cards (comma separated, e.g., '2с, Ah, 9s, ..')",
                placeholder="Leave empty if unknown"
            )
            opponent_cards_input = gr.Textbox(
                label="Known Opponent Cards (optional, format: 'As, 9h; Kd, Qd' for multiple opponents)",
                placeholder="Leave empty if unknown"
            )
            num_opponents_input = gr.Slider(
                minimum=1, maximum=9, value=3, step=1, 
                label="Number of Opponents"
            )
            num_simulations_input = gr.Radio(
                choices=[100, 1000, 5000, 10000],
                value=1000,
                label="Number of Simulations",
                type="value"
            )            
            calculate_button = gr.Button("Calculate Probabilities")
    
    # Results section (moved below the button)
    with gr.Row():
        with gr.Column():
            results_output = gr.Textbox(label="Results")
            plot_output = gr.Image(label="Combination Statistics")
    
    # Instructions section
    gr.Markdown("""
    ## How to use
    
    1. Enter your cards separated by commas (e.g., "Kh, 9d")
    2. Enter the table cards separated by commas. Use empty positions for unknown cards (e.g., "2h,Ah, 9s, , ")
    3. Optionally enter known opponent cards in format "As,9h;Kd,Qd" for multiple opponents
    4. Set the number of opponents and simulations
    5. Click Calculate Probabilities
    
    Card notation: 
    - 2-9, T (ten), J (jack), Q (queen), K (king), A (ace)
    - s (spades), h (hearts), d (diamonds), c (clubs)
    """)
    
    calculate_button.click(
        calculate_probabilities,
        inputs=[
            my_cards_input,
            table_cards_input, 
            opponent_cards_input,
            num_opponents_input,
            num_simulations_input
        ],
        outputs=[results_output, plot_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
