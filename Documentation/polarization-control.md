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

### Wave Plates  [^3]

A wave plate, also known as a retarder, is a birefrigent material that can be used to convert linearly polarized light to circularly polarized, or vice-versa. This happens by creating a phase difference between the vertical and horizontal components of the polarization.

Light travels more slowly in a medium with index of refraction n, given by $$c_{medium} = \frac{c_{vacuum}}{n}$$. A birefringent material has different indices of refraction over its vertical and horizontal axes. This means that if we have a vertical index of refraction $n_1$ and a horizontal index of refraction $n_2$, then the difference in velocities for the two components of the wave will be $$\Delta v = \frac{c}{n_1} - \frac{c}{n_2}$$ This means that, for a plate of thickness $d$, the phase difference (or retardance) experienced by the two components will be  $$\Delta \phi = 2 \pi d (\frac{n_1 - n_2}{\lambda})$$ Note the wavelength dependence. This means that we can tune the parameters $d$, $n_1$, and $n_2$ to get the phase difference we desire. For reference, a 90 degree phase difference will correspond to circularly polarized light.

It's important to note that a single wave plate is capable of converting circularly polarized light to linearly polarized, or vice versa. That means with two wave plates with the correct specifications, linearly polarized light can be converted to another arbitrary linear polarization with no signal lost. However, the dimensions and birefringence of the waveplates then need to be tuned to the specific application and wavelength.

### Liquid Crystal Variable Retarders (LCVRs)

A Liquid Crystal Variable Retarder (LCVR) is an optical componant with a retardance that is tunable via an input voltage. Since the LCVR is made of a liquid crystal material, where the individual molecules have a dipole moment, an applied AC voltage will cause the molecules to to angle themselves, with a higher voltage corresponding to a higher rotation. This in turn alters the birefringence of the material, and the retardance as a result. Here, we are using the LCVRs from Meadowlark Optics, [^4] which have a retardance v. input voltage at 632.8 nm of the following form

<p align="center">
  <img src="https://github.com/ubsuny/ML_LCVR-CP2P2024/assets/94491866/d70d3735-3cae-4929-9783-db2b17f0f090" />
</p>

Notably, this behavior is not only wavelength dependent, but very nonlinear for each individual wavelength.

Most importantly, a tunable output retardance means that an LCVR essentially functions as a tunable waveplate. One can potentially set any arbitrary polarization for any wavelength with two LCVRs. The only caveat is that LCVRs are sensitive and have a highly nonlinear response, so attaining the desired output polarization can take hours of calibration for a single polarization if performed manually.

## Machine Learning for Fast LCVR Calibration

To allow for switching polarizations quickly and reliably, we aim to implement a machine learning algorithm that takes training data on a small number of wavelengths, then creates a model that describes the output polarization response for all wavelengths in which the LCVR is usable. From here, a web interface allows for quick polarization switching where the only known parameter necessary is the wavelength of the input light.

To measure the output polarization from the LCVRs, a differential signal is measured from two photodiodes placed in front of a Wollaston prism that receives the output beam from the LCVRs. The Wollaston prism splits the light into two perpendicularly polarized components, so each photodiode measures an orthogonal component of the polarization. (Diagram to come). This gives us a signal that is robust to noise and measures the difference in signal between the two polarization components, which is easily back calculated into a polarization.

Training data is measured via a sweep over input LCVR voltages in the range of highest response. The
```Python
get_training_data()
```
function in lcvr_learning.py handles this process. After measuring several wavlengths, this data is fed into a neural network (unfinalized) which models the data and allows the polarization to be arbitrarily selected.


References:
[^1]: And1mu, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons
[^2]: Dave3457, Public domain, via Wikimedia Commons
[^3]: https://phys.libretexts.org/Bookshelves/Waves_and_Acoustics/The_Physics_of_Waves_(Goergi)/12%3A_Polarization/12.03%3A_Wave_Plates_and_Polarizers
[^4]: https://www.meadowlark.com/liquid-crystal-variable-retarder/
