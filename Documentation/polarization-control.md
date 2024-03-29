# Polarization Control with Liquid Crystal Variable Retarders

## Polarization of a Coherent Light Source

#### Linear Polarization
It's a well known fact that, classically speaking, light travels as a wave. More specifically, it will travel as a transverse electro-magnetic wave. This means that the propogation direction, the direction of the electric field, and the direction of the magnetic field of the wave are all mutually orthogonal. [^1]

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/EM-Wave.gif" />
</p>

The direction that the electric field points for such a wave is commonly known as the polarization of the wave. This polarization is typically described as an angle between 0 and 90 degrees, representing how much the electric field is "rotated" from the vertical. Coherent light sources such as lasers have a constant, measurable polarization. Due to this, we can also tune the polarization of the source through various means.

### Circular Polarization

We can also have a polarization that predicably rotates in time and space, known as circular polarization (see the figure below). [^2] 

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/d/d1/Circular.Polarization.Circularly.Polarized.Light_Left.Hand.Animation.305x190.255Colors.gif" />
</p>

This polarization is unique because the linear polarization is generally unable to be measured directly, needing more advanced techniques to analyse. It's also worth noting that individual photons carry only a circular polarization, so for non-classical applications circular polarization must be considered.

## Polarization-Controlling Optical Components

### Polarizers

The simplest method of tuning the linear polarization of a light source is through a polarizer. Polarizers act as a "screen" for polarization states, letting only the chosen direction of polarization through. These components are extremely limited however, as they both reduce the intensity of the incident light (potentially to zero!) and can not be used to meaningfully alter the state of circularly polarized light. In general, more advanced components must be used.

### Wave Plates [^3]

A wave plate, also known as a retarder, is a birefrigent material that can be used to convert linearly polarized light to circularly polarized, or vice-versa. This happens by creating a phase difference between the vertical and horizontal components of the polarization.

Light travels more slowly in a medium with index of refraction n, given by $$c_{medium} = \frac{c_{vacuum}}{n}$$. A birefringent material has different indices of refraction over its vertical and horizontal axes. This means that if we have a vertical index of refraction $n_1$ and a horizontal index of refraction $n_2$, then the difference in velocities for the two components of the wave will be $$\Delta v = \frac{c}{n_1} - \frac{c}{n_2}$$ This means that, for a plate of thickness $d$, the phase difference experienced by the two components will be  $$\Delta \phi = d (\frac{n_1 - n_2}{\lambda})$$ Note the wavelength dependence.

References:
[^1]: And1mu, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons
[^2]: Dave3457, Public domain, via Wikimedia Commons
[^3}: https://phys.libretexts.org/Bookshelves/Waves_and_Acoustics/The_Physics_of_Waves_(Goergi)/12%3A_Polarization/12.03%3A_Wave_Plates_and_Polarizers
