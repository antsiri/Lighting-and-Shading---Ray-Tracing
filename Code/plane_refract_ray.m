function [ray_air] = plane_refract_ray(y, slope, thickness, n, z)
theta1 = atan(slope);
theta2 = asin(n*sin(theta1));
slope2 = tan(theta2);
ray_air = (z-thickness)*slope2 + y;
end