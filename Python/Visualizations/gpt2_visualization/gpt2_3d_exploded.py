from manim import *

class GPT2_3D_Architecture(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        title = Text("GPT-2 Small Architecture", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        # Embedding layer
        embed = Prism(dimensions=[4,0.3,2], fill_color=BLUE, fill_opacity=0.6)
        embed.move_to(UP*2)

        embed_text = Text("Embedding\n(50257 x 768)", font_size=24)
        embed_text.move_to(embed.get_center())

        self.add(embed)
        self.add_fixed_in_frame_mobjects(embed_text)

        # Create transformer blocks
        layers = VGroup()

        for i in range(12):

            layer = Prism(
                dimensions=[4,0.3,2],
                fill_color=GRAY,
                fill_opacity=0.25
            )

            layer.shift(DOWN*0.7*i)

            layers.add(layer)

        self.play(Create(layers))

        label = Text("12 Transformer Blocks", font_size=28)
        label.to_edge(RIGHT)
        self.add_fixed_in_frame_mobjects(label)

        self.wait(2)

        self.begin_ambient_camera_rotation(rate=0.1)

        self.wait(4)

        self.stop_ambient_camera_rotation()



class TransformerBlockExploded(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(phi=70*DEGREES, theta=-45*DEGREES)

        title = Text("Transformer Block", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        # Attention layer
        attn = Prism(
            dimensions=[4,0.4,2],
            fill_color=BLUE,
            fill_opacity=0.5
        )

        attn.shift(UP)

        # MLP layer
        mlp = Prism(
            dimensions=[4,0.4,2],
            fill_color=PURPLE,
            fill_opacity=0.5
        )

        mlp.shift(DOWN)

        self.play(Create(attn), Create(mlp))

        attn_text = Text("Multi-Head Attention", font_size=28)
        mlp_text = Text("MLP", font_size=28)

        attn_text.move_to(attn.get_center())
        mlp_text.move_to(mlp.get_center())

        self.add_fixed_in_frame_mobjects(attn_text, mlp_text)

        self.wait(2)

        self.begin_ambient_camera_rotation(rate=0.1)

        self.wait(4)



class MLP_Weights_Visualization(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(phi=70*DEGREES, theta=-40*DEGREES)

        title = Text("MLP Layers (Compressed)", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        # c_fc layer
        fc = Prism(
            dimensions=[4,0.4,2],
            fill_color=GREEN,
            fill_opacity=0.5
        )

        fc.shift(UP)

        fc_label = Text("c_fc (768 → 3072)", font_size=28)
        fc_label.move_to(fc.get_center())

        # c_proj layer
        proj = Prism(
            dimensions=[4,0.4,2],
            fill_color=GREEN,
            fill_opacity=0.5
        )

        proj.shift(DOWN)

        proj_label = Text("c_proj (3072 → 768)", font_size=28)
        proj_label.move_to(proj.get_center())

        self.play(Create(fc), Create(proj))

        self.add_fixed_in_frame_mobjects(fc_label, proj_label)

        self.wait(2)

        arrow = Arrow3D(fc.get_bottom(), proj.get_top())

        self.play(Create(arrow))

        self.wait(2)



class SVD_Compression_Animation(Scene):

    def construct(self):

        title = Text("SVD Compression", font_size=48)
        title.to_edge(UP)

        self.play(Write(title))

        W = Matrix([["768","..."],["...","3072"]])

        label = Text("Weight Matrix W")

        group = VGroup(label, W).arrange(DOWN)

        self.play(Write(group))

        self.wait()

        svd = MathTex("W = U \\Sigma V^T", font_size=60)

        self.play(Transform(group, svd))

        self.wait(2)