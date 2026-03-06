from manim import *

class GPT2Architecture(Scene):
    def construct(self):

        title = Text("GPT-2 Small Architecture", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Embedding blocks
        token = Rectangle(width=2, height=0.8, color=BLUE)
        token_text = Text("Token Embedding\n(50257 x 768)", font_size=24).move_to(token)

        pos = Rectangle(width=2, height=0.8, color=BLUE)
        pos_text = Text("Position Embedding\n(1024 x 768)", font_size=24).move_to(pos)

        embed_group = VGroup(token, pos).arrange(DOWN, buff=0.5)
        embed_text = VGroup(token_text, pos_text)

        self.play(Create(embed_group), Write(embed_text))
        self.wait(1)

        embed = VGroup(embed_group, embed_text)

        # Transformer stack
        blocks = VGroup()

        for i in range(12):
            block = Rectangle(width=3, height=0.5, color=GREEN)
            label = Text(f"Transformer Block {i}", font_size=20).move_to(block)
            blocks.add(VGroup(block, label))

        blocks.arrange(DOWN, buff=0.2)
        blocks.next_to(embed, DOWN, buff=1)

        stack_label = Text("12 Transformer Layers", font_size=28)
        stack_label.next_to(blocks, RIGHT)

        self.play(Create(blocks), Write(stack_label))
        self.wait(2)

        # Output head
        head = Rectangle(width=3, height=0.8, color=ORANGE)
        head_text = Text("LM Head\n(768 → 50257)", font_size=24).move_to(head)

        head.next_to(blocks, DOWN, buff=1)

        self.play(Create(head), Write(head_text))
        self.wait(2)


class TransformerBlockExploded(Scene):
    def construct(self):

        title = Text("GPT-2 Transformer Block", font_size=48)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        block = Rectangle(width=5, height=3, color=GREEN)

        attn = Rectangle(width=2, height=1, color=BLUE)
        attn_text = Text("Self Attention", font_size=24).move_to(attn)

        mlp = Rectangle(width=2, height=1, color=PURPLE)
        mlp_text = Text("MLP", font_size=24).move_to(mlp)

        components = VGroup(
            VGroup(attn, attn_text),
            VGroup(mlp, mlp_text)
        ).arrange(DOWN, buff=0.7)

        self.play(Create(block))
        self.play(components.animate.move_to(block))
        self.wait()

        self.play(FadeOut(block))
        self.wait()

        self.play(components.animate.arrange(RIGHT, buff=2))
        self.wait(2)


class MLPCompression(Scene):
    def construct(self):

        title = Text("MLP Layer Compression", font_size=48)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        # Original layers
        c_fc = Rectangle(width=3, height=1, color=BLUE)
        fc_text = Text("c_fc\n768 → 3072", font_size=24).move_to(c_fc)

        c_proj = Rectangle(width=3, height=1, color=BLUE)
        proj_text = Text("c_proj\n3072 → 768", font_size=24).move_to(c_proj)

        mlp = VGroup(
            VGroup(c_fc, fc_text),
            VGroup(c_proj, proj_text)
        ).arrange(DOWN, buff=1)

        self.play(Create(mlp))
        self.wait(2)

        compression = Text(
            "Apply SVD / Kronecker Compression",
            font_size=32,
            color=YELLOW
        )

        compression.next_to(mlp, DOWN)

        self.play(Write(compression))
        self.wait(2)

        # Exploded compressed layers
        u = Rectangle(width=1.5, height=0.8, color=RED)
        s = Rectangle(width=1.5, height=0.8, color=RED)
        v = Rectangle(width=1.5, height=0.8, color=RED)

        u_text = Text("U", font_size=20).move_to(u)
        s_text = Text("S", font_size=20).move_to(s)
        v_text = Text("Vᵀ", font_size=20).move_to(v)

        svd = VGroup(
            VGroup(u, u_text),
            VGroup(s, s_text),
            VGroup(v, v_text)
        ).arrange(RIGHT, buff=0.5)

        svd.next_to(c_fc, RIGHT, buff=2)

        arrow = Arrow(c_fc.get_right(), svd.get_left())

        self.play(Create(arrow), Create(svd))
        self.wait(3)